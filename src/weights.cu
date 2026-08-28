// Real-model weight loading: host dequant + transpose into fp16, pack into the
// streaming blob layout, upload non-layer weights to device.
#include "weights.h"
#include "kquant.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <thread>
#include <atomic>
#include <algorithm>

namespace {

// Dequant n elements of a GGUF tensor into fp16 (host). F32/F16/Q8_0.
bool dequant_host(const uint8_t* data, uint32_t type, size_t n, __half* dst) {
    if (type == GGML_F32) {
        const float* f = reinterpret_cast<const float*>(data);
        for (size_t i = 0; i < n; ++i) dst[i] = __float2half(f[i]);
        return true;
    }
    if (type == GGML_F16) {
        memcpy(dst, data, n * sizeof(__half));
        return true;
    }
    if (type == GGML_Q8_0) {
        const size_t nb = n / 32;
        const uint8_t* p = data;
        for (size_t b = 0; b < nb; ++b) {
            __half d; memcpy(&d, p, 2);
            float df = __half2float(d);
            const int8_t* qs = reinterpret_cast<const int8_t*>(p + 2);
            for (int j = 0; j < 32; ++j) dst[b * 32 + j] = __float2half(df * (float)qs[j]);
            p += 34;
        }
        return true;
    }
    if (type == GGML_Q4_K || type == GGML_Q6_K) {
        std::vector<float> f(n);
        if (type == GGML_Q4_K) dequant_q4_K_f32(data, n, f.data());
        else dequant_q6_K_f32(data, n, f.data());
        for (size_t i = 0; i < n; ++i) dst[i] = __float2half(f[i]);
        return true;
    }
    return false;  // other quants not yet supported
}

// Quantize an fp16 array to Q8_0 (34 bytes / 32 values): d = amax/127, q = round(x/d).
void quantize_q8_0(const __half* x, size_t n, uint8_t* out) {
    const size_t nb = n / 32;
    for (size_t b = 0; b < nb; ++b) {
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) amax = fmaxf(amax, fabsf(__half2float(x[b * 32 + j])));
        const float d = amax / 127.0f;
        const float id = d > 0.0f ? 1.0f / d : 0.0f;
        __half dh = __float2half(d);
        memcpy(out, &dh, 2);
        int8_t* qs = reinterpret_cast<int8_t*>(out + 2);
        for (int j = 0; j < 32; ++j) {
            int q = (int)lroundf(__half2float(x[b * 32 + j]) * id);
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            qs[j] = (int8_t)q;
        }
        out += 34;
    }
}

// Quantize fp16 -> Q4_0 (18 bytes / 32 values): d = max/-8; nibble j = round(x/d)+8,
// packed low=elem j, high=elem j+16. x = d*(q-8).
void quantize_q4_0(const __half* x, size_t n, uint8_t* out) {
    const size_t nb = n / 32;
    for (size_t b = 0; b < nb; ++b) {
        float amax = 0.0f, vmax = 0.0f;
        for (int j = 0; j < 32; ++j) {
            float v = __half2float(x[b * 32 + j]);
            if (fabsf(v) > amax) { amax = fabsf(v); vmax = v; }
        }
        const float d = vmax / -8.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        __half dh = __float2half(d);
        memcpy(out, &dh, 2);
        uint8_t* qs = out + 2;
        for (int j = 0; j < 16; ++j) {
            int q0 = (int)(__half2float(x[b * 32 + j]) * id + 8.5f);
            int q1 = (int)(__half2float(x[b * 32 + j + 16]) * id + 8.5f);
            q0 = q0 < 0 ? 0 : (q0 > 15 ? 15 : q0);
            q1 = q1 < 0 ? 0 : (q1 > 15 ? 15 : q1);
            qs[j] = (uint8_t)(q0 | (q1 << 4));
        }
        out += 18;
    }
}

// Transpose src[rows, cols] (row-major) -> dst[cols, rows] (row-major), fp16.
void transpose_host(const __half* src, __half* dst, int rows, int cols) {
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            dst[(size_t)c * rows + r] = src[(size_t)r * cols + c];
}

// GGUF stores a weight matrix as [out, in] row-major (ne0=in, ne1=out). Return
// out and in from the tensor dims.
void weight_out_in(const TensorInfo& t, int* out, int* in) {
    *in = (int)t.dims[0];
    *out = t.dims.size() > 1 ? (int)t.dims[1] : 1;
}

// Dequant a tensor into a temp fp16 buffer (returns element count).
bool dequant_tensor(const GgufFile& g, const TensorInfo& t, std::vector<__half>& tmp) {
    size_t n = gguf_tensor_elements(t);
    tmp.resize(n);
    return dequant_host(gguf_tensor_data(g, t), t.ggml_type, n, tmp.data());
}

// Upload a dequantized tensor to a fresh device buffer.
__half* upload_tensor(const GgufFile& g, const TensorInfo& t, std::string* err) {
    std::vector<__half> tmp;
    if (!dequant_tensor(g, t, tmp)) { if (err) *err = "unsupported quant for " + t.name; return nullptr; }
    __half* d;
    cudaMalloc(&d, tmp.size() * sizeof(__half));
    cudaMemcpy(d, tmp.data(), tmp.size() * sizeof(__half), cudaMemcpyHostToDevice);
    return d;
}

}  // namespace

bool load_model(const GgufFile& g, LoadedModel* out, int q_bits, std::string* err) {
    out->q_bits = (q_bits == 4) ? 4 : 8;
    ModelConfig mc;
    if (!read_config(g, &mc, err)) return false;
    out->arch = mc.arch;
    out->n_layers = mc.n_layers;
    out->cfg = LlamaConfig{mc.dim, mc.n_heads, mc.n_kv_heads, mc.head_dim, mc.ffn_dim,
                           mc.eps, mc.rope_base};
    const LlamaConfig& cfg = out->cfg;

    const TensorInfo* embd = gguf_find_tensor(g, "token_embd.weight");
    if (!embd) { if (err) *err = "missing token_embd.weight"; return false; }
    out->vocab = embd->dims.size() > 1 ? (int)embd->dims[1] : (int)(gguf_tensor_elements(*embd) / cfg.dim);

    out->blob = layer_blob_layout(cfg);
    const size_t per = out->blob.total_elems;
    out->q8_layer_bytes = (per / 32) * (out->q_bits == 4 ? 18 : 34);
    cudaHostAlloc((void**)&out->h_layer_q8, out->q8_layer_bytes * out->n_layers, cudaHostAllocDefault);

    // Each layer is independent -> load them in parallel across CPU cores.
    // dequant [out,in] -> transpose -> [in,out] -> re-quantize to Q8_0. Per-thread
    // temp buffers; threads write disjoint layers of the pinned Q8_0 array.
    std::atomic<int> failed{-1};
    auto load_layer = [&](int L) {
        std::vector<__half> base(per), tmp, tt;
        LayerTensors lt; std::string e;
        if (!get_layer_tensors(g, L, &lt, &e)) { failed = L; return; }
        if (!dequant_tensor(g, *lt.attn_norm, tmp)) { failed = L; return; }
        memcpy(base.data() + out->blob.off_attn_norm, tmp.data(), tmp.size() * sizeof(__half));
        if (!dequant_tensor(g, *lt.ffn_norm, tmp)) { failed = L; return; }
        memcpy(base.data() + out->blob.off_ffn_norm, tmp.data(), tmp.size() * sizeof(__half));

        const TensorInfo* pt[7] = {lt.wq, lt.wk, lt.wv, lt.wo, lt.wgate, lt.wup, lt.wdown};
        size_t poff[7] = {out->blob.off_wq, out->blob.off_wk, out->blob.off_wv, out->blob.off_wo,
                          out->blob.off_wgate, out->blob.off_wup, out->blob.off_wdown};
        for (int i = 0; i < 7; ++i) {
            if (!dequant_tensor(g, *pt[i], tmp)) { failed = L; return; }
            int o, in; weight_out_in(*pt[i], &o, &in);
            tt.resize((size_t)o * in);
            transpose_host(tmp.data(), tt.data(), o, in);   // [out,in] -> [in,out]
            memcpy(base.data() + poff[i], tt.data(), tt.size() * sizeof(__half));
        }
        uint8_t* dst = out->h_layer_q8 + (size_t)L * out->q8_layer_bytes;
        if (out->q_bits == 4) quantize_q4_0(base.data(), per, dst);
        else quantize_q8_0(base.data(), per, dst);
    };

    unsigned hw = std::thread::hardware_concurrency();
    int nthreads = (int)std::min<unsigned>(hw ? hw : 4u, (unsigned)out->n_layers);
    printf("loading %d layers on %d threads...\n", out->n_layers, nthreads);
    std::vector<std::thread> threads;
    for (int t = 0; t < nthreads; ++t)
        threads.emplace_back([&, t]() { for (int L = t; L < out->n_layers; L += nthreads) load_layer(L); });
    for (auto& th : threads) th.join();
    if (failed.load() >= 0) { if (err) *err = "failed to load a layer (unsupported quant?)"; return false; }

    // Non-layer weights on device (no transpose: token_embd row=id, output row=vocab).
    out->rw.token_embd = upload_tensor(g, *embd, err);
    if (!out->rw.token_embd) return false;

    const TensorInfo* fn = gguf_find_tensor(g, "output_norm.weight");
    if (!fn) { if (err) *err = "missing output_norm.weight"; return false; }
    out->rw.final_norm = upload_tensor(g, *fn, err);
    if (!out->rw.final_norm) return false;

    const TensorInfo* ow = gguf_find_tensor(g, "output.weight");
    out->rw.output = ow ? upload_tensor(g, *ow, err) : out->rw.token_embd;  // tied fallback
    if (!out->rw.output) return false;

    return true;
}

void free_model(LoadedModel* m) {
    if (m->h_layer_q8) cudaFreeHost(m->h_layer_q8);
    if (m->rw.token_embd) cudaFree((void*)m->rw.token_embd);
    if (m->rw.final_norm) cudaFree((void*)m->rw.final_norm);
    if (m->rw.output && m->rw.output != m->rw.token_embd) cudaFree((void*)m->rw.output);
    m->h_layer_q8 = nullptr;
}

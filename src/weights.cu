// Real-model weight loading: host dequant + transpose into fp16, pack into the
// streaming blob layout, upload non-layer weights to device.
#include "weights.h"
#include "kquant.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

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

bool load_model(const GgufFile& g, LoadedModel* out, std::string* err) {
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
    out->q8_layer_bytes = (per / 32) * 34;
    cudaHostAlloc((void**)&out->h_layer_q8, out->q8_layer_bytes * out->n_layers, cudaHostAllocDefault);

    std::vector<__half> layer_buf(per), tmp, tt;
    const char* proj[7] = {"attn_q.weight", "attn_k.weight", "attn_v.weight",
                           "attn_output.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"};

    for (int L = 0; L < out->n_layers; ++L) {
        LayerTensors lt;
        if (!get_layer_tensors(g, L, &lt, err)) return false;
        __half* base = layer_buf.data();   // fp16 temp for this layer

        // norms (1D, no transpose)
        if (!dequant_tensor(g, *lt.attn_norm, tmp)) { if (err) *err = "attn_norm quant"; return false; }
        memcpy(base + out->blob.off_attn_norm, tmp.data(), tmp.size() * sizeof(__half));
        if (!dequant_tensor(g, *lt.ffn_norm, tmp)) { if (err) *err = "ffn_norm quant"; return false; }
        memcpy(base + out->blob.off_ffn_norm, tmp.data(), tmp.size() * sizeof(__half));

        // projections: dequant [out,in] then transpose -> [in,out] at blob offset
        const TensorInfo* pt[7] = {lt.wq, lt.wk, lt.wv, lt.wo, lt.wgate, lt.wup, lt.wdown};
        size_t poff[7] = {out->blob.off_wq, out->blob.off_wk, out->blob.off_wv, out->blob.off_wo,
                          out->blob.off_wgate, out->blob.off_wup, out->blob.off_wdown};
        for (int i = 0; i < 7; ++i) {
            if (!dequant_tensor(g, *pt[i], tmp)) { if (err) *err = std::string("quant ") + proj[i]; return false; }
            int o, in; weight_out_in(*pt[i], &o, &in);
            tt.resize((size_t)o * in);
            transpose_host(tmp.data(), tt.data(), o, in);   // [out,in] -> [in,out]
            memcpy(base + poff[i], tt.data(), tt.size() * sizeof(__half));
        }
        // Re-quantize the fp16 layer to Q8_0 for streaming (half the RAM + transfer).
        quantize_q8_0(base, per, out->h_layer_q8 + (size_t)L * out->q8_layer_bytes);
        if (L == 0 || (L + 1) % 8 == 0) printf("  loaded layer %d/%d\n", L + 1, out->n_layers);
    }

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

// MoE model loading + generation (Mixtral-style).
#include "moe_model.h"
#include "weights_util.h"
#include "dequant.h"
#include "rmsnorm.h"
#include "runner.h"
#include "sampling.h"
#include "gemm.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <cuda_runtime.h>

namespace {

void block_info(uint32_t type, size_t* be, size_t* bb) {
    switch (type) {
        case GGML_F32:  *be = 1;   *bb = 4;   break;
        case GGML_F16:  *be = 1;   *bb = 2;   break;
        case GGML_Q8_0: *be = 32;  *bb = 34;  break;
        case GGML_Q4_0: *be = 32;  *bb = 18;  break;
        case GGML_Q4_K: *be = 256; *bb = 144; break;
        case GGML_Q6_K: *be = 256; *bb = 210; break;
        default:        *be = 0;   *bb = 0;   break;
    }
}

// Dequant the chunk starting at element elem_offset (n elements) of a quantized tensor.
bool dequant_chunk(const GgufFile& g, const TensorInfo& t, size_t elem_offset, size_t n, __half* dst) {
    size_t be, bb; block_info(t.ggml_type, &be, &bb);
    if (be == 0) return false;
    size_t byte_off = (elem_offset / be) * bb;
    return dequant_host(gguf_tensor_data(g, t) + byte_off, t.ggml_type, n, dst);
}

std::string mk(const std::string& arch, const char* s) { return arch + "." + s; }

bool read_moe_config(const GgufFile& g, LoadedMoeModel* out, std::string* err) {
    if (!gguf_get_str(g, "general.architecture", &out->arch)) { if (err) *err = "no arch"; return false; }
    const std::string& a = out->arch;
    uint32_t u; float f;
    auto need = [&](const char* k, int* dst) {
        if (!gguf_get_u32(g, mk(a, k), &u)) return false; *dst = (int)u; return true; };
    if (!need("block_count", &out->n_layers)) { if (err) *err = "no block_count"; return false; }
    if (!need("embedding_length", &out->cfg.dim)) { if (err) *err = "no embedding_length"; return false; }
    if (!need("attention.head_count", &out->cfg.n_heads)) { if (err) *err = "no head_count"; return false; }
    if (gguf_get_u32(g, mk(a, "attention.head_count_kv"), &u)) out->cfg.n_kv_heads = (int)u;
    else out->cfg.n_kv_heads = out->cfg.n_heads;
    if (!need("expert_count", &out->mcfg.n_experts)) { if (err) *err = "no expert_count"; return false; }
    if (!need("expert_used_count", &out->mcfg.n_used)) { if (err) *err = "no expert_used_count"; return false; }
    if (!need("expert_feed_forward_length", &out->mcfg.expert_ffn)) { if (err) *err = "no expert_ffn"; return false; }
    out->cfg.head_dim = out->cfg.dim / out->cfg.n_heads;
    out->cfg.ffn_dim = 0;
    out->cfg.eps = 1e-5f; out->cfg.rope_base = 10000.0f;
    if (gguf_get_f32(g, mk(a, "attention.layer_norm_rms_epsilon"), &f)) out->cfg.eps = f;
    if (gguf_get_f32(g, mk(a, "rope.freq_base"), &f)) out->cfg.rope_base = f;
    out->mcfg.has_shared = false;
    return true;
}

MoeLayerBlob layout(const LlamaConfig& c, const MoeConfig& mc) {
    size_t dim = c.dim, qd = (size_t)c.n_heads*c.head_dim, kvd = (size_t)c.n_kv_heads*c.head_dim;
    size_t ef = mc.expert_ffn, E = mc.n_experts;
    MoeLayerBlob b;
    b.off_attn_norm = 0;
    b.off_wq        = b.off_attn_norm + dim;
    b.off_wk        = b.off_wq + dim*qd;
    b.off_wv        = b.off_wk + dim*kvd;
    b.off_wo        = b.off_wv + dim*kvd;
    b.off_ffn_norm  = b.off_wo + qd*dim;
    b.off_router    = b.off_ffn_norm + dim;
    b.off_experts   = b.off_router + dim*E;
    b.off_egate = 0; b.off_eup = dim*ef; b.off_edown = 2*dim*ef;
    b.expert_stride = 3*dim*ef;
    b.total_elems   = b.off_experts + E*b.expert_stride;
    return b;
}

// Dequant [out,in] tensor (or chunk) into tmp, transpose -> [in,out] at base+off.
bool load_mat(const GgufFile& g, const TensorInfo* t, size_t chunk_elem_off, int o, int in,
              __half* base, size_t off, std::vector<__half>& tmp, std::vector<__half>& tt) {
    tmp.resize((size_t)o*in);
    if (!dequant_chunk(g, *t, chunk_elem_off, (size_t)o*in, tmp.data())) return false;
    tt.resize((size_t)o*in);
    transpose_host(tmp.data(), tt.data(), o, in);
    memcpy(base + off, tt.data(), (size_t)o*in*sizeof(__half));
    return true;
}

}  // namespace

bool is_moe_model(const GgufFile& g) {
    std::string a; if (!gguf_get_str(g, "general.architecture", &a)) return false;
    uint32_t u; return gguf_get_u32(g, a + ".expert_count", &u) && u > 1;
}

bool load_moe_model(const GgufFile& g, LoadedMoeModel* out, int q_bits, std::string* err) {
    out->q_bits = (q_bits == 4) ? 4 : 8;
    if (!read_moe_config(g, out, err)) return false;
    const LlamaConfig& cfg = out->cfg; const MoeConfig& mc = out->mcfg;
    int dim = cfg.dim, qd = cfg.n_heads*cfg.head_dim, kvd = cfg.n_kv_heads*cfg.head_dim, ef = mc.expert_ffn, E = mc.n_experts;

    const TensorInfo* embd = gguf_find_tensor(g, "token_embd.weight");
    if (!embd) { if (err) *err = "no token_embd"; return false; }
    out->vocab = embd->dims.size() > 1 ? (int)embd->dims[1] : (int)(gguf_tensor_elements(*embd)/dim);

    out->blob = layout(cfg, mc);
    const size_t per = out->blob.total_elems;
    out->q_layer_bytes = (per/32) * (out->q_bits == 4 ? 18 : 34);
    cudaHostAlloc((void**)&out->h_layer_q, out->q_layer_bytes * out->n_layers, cudaHostAllocDefault);

    std::atomic<int> failed{-1};
    auto load_layer = [&](int L) {
        std::vector<__half> base(per), tmp, tt;
        char name[96];
        auto T = [&](const char* suf) -> const TensorInfo* {
            snprintf(name, sizeof(name), "blk.%d.%s", L, suf); return gguf_find_tensor(g, name); };
        const TensorInfo *an = T("attn_norm.weight"), *fn = T("ffn_norm.weight");
        const TensorInfo *wq = T("attn_q.weight"), *wk = T("attn_k.weight"), *wv = T("attn_v.weight"), *wo = T("attn_output.weight");
        const TensorInfo *ri = T("ffn_gate_inp.weight");
        const TensorInfo *ge = T("ffn_gate_exps.weight"), *ue = T("ffn_up_exps.weight"), *de = T("ffn_down_exps.weight");
        if (!an||!fn||!wq||!wk||!wv||!wo||!ri||!ge||!ue||!de) { failed = L; return; }

        if (!dequant_tensor(g, *an, tmp)) { failed = L; return; }
        memcpy(base.data()+out->blob.off_attn_norm, tmp.data(), tmp.size()*sizeof(__half));
        if (!dequant_tensor(g, *fn, tmp)) { failed = L; return; }
        memcpy(base.data()+out->blob.off_ffn_norm, tmp.data(), tmp.size()*sizeof(__half));

        int o, in;
        weight_out_in(*wq, &o, &in); if (!load_mat(g, wq, 0, o, in, base.data(), out->blob.off_wq, tmp, tt)) { failed=L; return; }
        weight_out_in(*wk, &o, &in); if (!load_mat(g, wk, 0, o, in, base.data(), out->blob.off_wk, tmp, tt)) { failed=L; return; }
        weight_out_in(*wv, &o, &in); if (!load_mat(g, wv, 0, o, in, base.data(), out->blob.off_wv, tmp, tt)) { failed=L; return; }
        weight_out_in(*wo, &o, &in); if (!load_mat(g, wo, 0, o, in, base.data(), out->blob.off_wo, tmp, tt)) { failed=L; return; }
        // router [E, dim] -> transpose [dim, E]
        if (!load_mat(g, ri, 0, E, dim, base.data(), out->blob.off_router, tmp, tt)) { failed=L; return; }

        // experts: each e slice. gate/up: [ef, dim] per expert -> [dim, ef]. down: [dim, ef] -> [ef, dim].
        for (int e = 0; e < E; ++e) {
            size_t ebase = out->blob.off_experts + (size_t)e*out->blob.expert_stride;
            if (!load_mat(g, ge, (size_t)e*dim*ef, ef, dim, base.data(), ebase+out->blob.off_egate, tmp, tt)) { failed=L; return; }
            if (!load_mat(g, ue, (size_t)e*dim*ef, ef, dim, base.data(), ebase+out->blob.off_eup, tmp, tt)) { failed=L; return; }
            if (!load_mat(g, de, (size_t)e*ef*dim, dim, ef, base.data(), ebase+out->blob.off_edown, tmp, tt)) { failed=L; return; }
        }

        uint8_t* dst = out->h_layer_q + (size_t)L*out->q_layer_bytes;
        if (out->q_bits == 4) quantize_q4_0(base.data(), per, dst); else quantize_q8_0(base.data(), per, dst);
    };

    unsigned hw = std::thread::hardware_concurrency();
    int nt = (int)(hw ? (hw < (unsigned)out->n_layers ? hw : out->n_layers) : 4);
    printf("loading %d MoE layers (%d experts each) on %d threads...\n", out->n_layers, E, nt);
    std::vector<std::thread> th;
    for (int t = 0; t < nt; ++t) th.emplace_back([&, t]() { for (int L = t; L < out->n_layers; L += nt) load_layer(L); });
    for (auto& x : th) x.join();
    if (failed.load() >= 0) { if (err) *err = "failed loading a MoE layer"; return false; }

    // non-layer weights (reuse dense loader pattern via a temp LoadedModel would duplicate;
    // just dequant here).
    std::vector<__half> tmp;
    auto up = [&](const TensorInfo* t) -> __half* {
        if (!dequant_tensor(g, *t, tmp)) return nullptr;
        __half* d; cudaMalloc(&d, tmp.size()*sizeof(__half));
        cudaMemcpy(d, tmp.data(), tmp.size()*sizeof(__half), cudaMemcpyHostToDevice); return d; };
    out->rw.token_embd = up(embd);
    const TensorInfo* on = gguf_find_tensor(g, "output_norm.weight");
    const TensorInfo* ow = gguf_find_tensor(g, "output.weight");
    if (!on) { if (err) *err = "no output_norm"; return false; }
    out->rw.final_norm = up(on);
    out->rw.output = ow ? up(ow) : out->rw.token_embd;
    if (!out->rw.token_embd || !out->rw.final_norm || !out->rw.output) { if (err) *err = "upload failed"; return false; }
    return true;
}

void free_moe_model(LoadedMoeModel* m) {
    if (m->h_layer_q) cudaFreeHost(m->h_layer_q);
    if (m->rw.token_embd) cudaFree((void*)m->rw.token_embd);
    if (m->rw.final_norm) cudaFree((void*)m->rw.final_norm);
    if (m->rw.output && m->rw.output != m->rw.token_embd) cudaFree((void*)m->rw.output);
    m->h_layer_q = nullptr;
}

bool moe_generate(LoadedMoeModel& m, std::vector<int>& ids, int n_gen,
                  int n_slots, int batch_layers, std::string* err) {
    const LlamaConfig& cfg = m.cfg; const MoeConfig& mc = m.mcfg;
    int dim = cfg.dim, qd = cfg.n_heads*cfg.head_dim, kvd = cfg.n_kv_heads*cfg.head_dim, ef = mc.expert_ffn, E = mc.n_experts;
    int prompt_len = (int)ids.size(), max_T = prompt_len + n_gen;
    size_t per = m.blob.total_elems;

    // Device buffers.
    __half *hidden, *normed, *arena, *logits_d; int* positions;
    cudaMalloc(&hidden, (size_t)max_T*dim*2); cudaMalloc(&normed, (size_t)max_T*dim*2);
    cudaMalloc(&arena, per*2); cudaMalloc(&logits_d, (size_t)m.vocab*2); cudaMalloc(&positions, max_T*4);
    LayerScratch s;
    cudaMalloc(&s.xn,(size_t)max_T*dim*2); cudaMalloc(&s.q,(size_t)max_T*qd*2);
    cudaMalloc(&s.k,(size_t)max_T*kvd*2); cudaMalloc(&s.v,(size_t)max_T*kvd*2);
    cudaMalloc(&s.att,(size_t)max_T*qd*2); cudaMalloc(&s.proj,(size_t)max_T*dim*2);
    cudaMalloc(&s.gate,(size_t)max_T*ef*2); cudaMalloc(&s.up,(size_t)max_T*ef*2);
    MoeScratch ms;
    cudaMalloc(&ms.logits,(size_t)max_T*E*2); cudaMalloc(&ms.route_w,(size_t)max_T*E*sizeof(float));
    cudaMalloc(&ms.gate,(size_t)max_T*ef*2); cudaMalloc(&ms.up,(size_t)max_T*ef*2);
    cudaMalloc(&ms.ye,(size_t)max_T*dim*2); cudaMalloc(&ms.moe_out,(size_t)max_T*dim*2);
    cudaHostAlloc((void**)&ms.h_route, (size_t)max_T*E*sizeof(float), cudaHostAllocDefault);
    KVCache kv; kv.n_layers=m.n_layers; kv.max_T=max_T; kv.kvd=kvd; kv.len=0;
    cudaMalloc(&kv.K,(size_t)m.n_layers*max_T*kvd*2); cudaMalloc(&kv.V,(size_t)m.n_layers*max_T*kvd*2);

    SlotPool pool; slotpool_create(&pool, n_slots, ((size_t)batch_layers*m.q_layer_bytes+1)/2);
    printf("streamed working set = %d slots x %.2f GB (Q%d) = %.2f GB VRAM\n",
           n_slots, m.q_layer_bytes/1e9, m.q_bits, (double)n_slots*batch_layers*m.q_layer_bytes/1e9);

    // Expert / attn weight views into the arena (arena is refreshed per layer by dequant).
    std::vector<const __half*> wg(E), wu(E), wd(E);
    for (int e = 0; e < E; ++e) {
        size_t eb = m.blob.off_experts + (size_t)e*m.blob.expert_stride;
        wg[e] = arena + eb + m.blob.off_egate; wu[e] = arena + eb + m.blob.off_eup; wd[e] = arena + eb + m.blob.off_edown;
    }
    MoeLayerWeights w;
    w.attn_norm=arena+m.blob.off_attn_norm; w.wq=arena+m.blob.off_wq; w.wk=arena+m.blob.off_wk;
    w.wv=arena+m.blob.off_wv; w.wo=arena+m.blob.off_wo; w.ffn_norm=arena+m.blob.off_ffn_norm;
    w.w_gate=arena+m.blob.off_router; w.wgate=wg.data(); w.wup=wu.data(); w.wdown=wd.data();

    int* d_ids; cudaMalloc(&d_ids, max_T*4);
    cudaStream_t cs, cm; cudaStreamCreate(&cs); cudaStreamCreate(&cm);
    Gemm gemm; gemm_create(&gemm);
    std::vector<__half> hl(m.vocab); std::vector<float> lf(m.vocab);
    int bpl = (int)(per/32);

    auto forward = [&](const int* d_tok, int n_new, int len_before) {
        std::vector<int> pos(n_new); for (int i=0;i<n_new;++i) pos[i]=len_before+i;
        cudaMemcpy(positions, pos.data(), n_new*4, cudaMemcpyHostToDevice);
        embed_tokens(m.rw.token_embd, d_tok, hidden, n_new, dim, cm);
        int NB = (m.n_layers + batch_layers - 1)/batch_layers;
        auto prefetch = [&](int b){ if(b>=NB) return; int sl=b%pool.n_slots; int f=b*batch_layers;
            int cnt=(f+batch_layers>m.n_layers)?(m.n_layers-f):batch_layers;
            cudaStreamWaitEvent(cs, pool.compute_done[sl], 0);
            cudaMemcpyAsync((uint8_t*)pool.d_slots[sl], m.h_layer_q+(size_t)f*m.q_layer_bytes,
                            (size_t)cnt*m.q_layer_bytes, cudaMemcpyHostToDevice, cs);
            cudaEventRecord(pool.copy_done[sl], cs); };
        prefetch(0);
        for (int b=0;b<NB;++b){ prefetch(b+1); int sl=b%pool.n_slots; int f=b*batch_layers;
            int cnt=(f+batch_layers>m.n_layers)?(m.n_layers-f):batch_layers;
            cudaStreamWaitEvent(cm, pool.copy_done[sl], 0);
            const uint8_t* slot=(const uint8_t*)pool.d_slots[sl];
            for (int l=0;l<cnt;++l){ const uint8_t* q=slot+(size_t)l*m.q_layer_bytes;
                if (m.q_bits==4) dequant_q4_0((const BlockQ40*)q, arena, bpl, cm);
                else dequant_q8_0((const BlockQ80*)q, arena, bpl, cm);
                moe_layer_forward_cached(cfg, mc, w, hidden, positions, n_new, f+l, len_before, kv, s, ms, &gemm, cm); }
            cudaEventRecord(pool.compute_done[sl], cm); }
        cudaStreamSynchronize(cm);
        rmsnorm(hidden, m.rw.final_norm, normed, n_new, dim, cfg.eps, cm);
        gemm_rowmajor(&gemm, m.rw.output, normed+(size_t)(n_new-1)*dim, logits_d, m.vocab, 1, dim, cm);
        cudaStreamSynchronize(cm);
    };
    auto sample = [&]() -> int {
        cudaMemcpy(hl.data(), logits_d, m.vocab*2, cudaMemcpyDeviceToHost);
        for (int v=0;v<m.vocab;++v) lf[v]=__half2float(hl[v]);
        return sample_greedy(lf.data(), m.vocab);
    };

    printf("prefill %d + generate %d (MoE, KV cache)...\n", prompt_len, n_gen);
    cudaMemcpy(d_ids, ids.data(), prompt_len*4, cudaMemcpyHostToDevice);
    forward(d_ids, prompt_len, 0); kv.len = prompt_len;
    int next = sample(); ids.push_back(next);
    auto t0 = std::chrono::steady_clock::now();
    for (int step=1; step<n_gen; ++step) {
        cudaMemcpy(d_ids, &next, 4, cudaMemcpyHostToDevice);
        forward(d_ids, 1, kv.len); kv.len += 1;
        next = sample(); ids.push_back(next);
    }
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { if (err) *err = cudaGetErrorString(e); return false; }
    if (n_gen > 1) printf("%.2f tokens/s decode (%d tokens, KV cache)\n", (n_gen-1)/secs, n_gen-1);

    cudaFree(kv.K); cudaFree(kv.V); slotpool_destroy(&pool); gemm_destroy(&gemm); free_moe_model(&m);
    return true;
}

// MoE model: stream original quantized weights, dequant + transpose on the GPU.
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
#include <chrono>
#include <cuda_runtime.h>

namespace {

void block_info(uint32_t type, size_t* be, size_t* bb) {
    switch (type) {
        case GGML_F32:  *be = 1;   *bb = 4;   break;
        case GGML_F16:  *be = 1;   *bb = 2;   break;
        case GGML_Q8_0: *be = 32;  *bb = 34;  break;
        case GGML_Q4_0: *be = 32;  *bb = 18;  break;
        case GGML_Q5_0: *be = 32;  *bb = 22;  break;
        case GGML_Q4_K: *be = 256; *bb = 144; break;
        case GGML_Q5_K: *be = 256; *bb = 176; break;
        case GGML_Q6_K: *be = 256; *bb = 210; break;
        default:        *be = 0;   *bb = 0;   break;
    }
}

std::string mk(const std::string& a, const char* s) { return a + "." + s; }

bool read_moe_config(const GgufFile& g, LoadedMoeModel* out, std::string* err) {
    if (!gguf_get_str(g, "general.architecture", &out->arch)) { if (err) *err = "no arch"; return false; }
    const std::string& a = out->arch;
    uint32_t u; float f;
    auto need = [&](const char* k, int* dst) { if (!gguf_get_u32(g, mk(a, k), &u)) return false; *dst = (int)u; return true; };
    if (!need("block_count", &out->n_layers)) { if (err) *err = "no block_count"; return false; }
    if (!need("embedding_length", &out->cfg.dim)) { if (err) *err = "no embedding_length"; return false; }
    if (!need("attention.head_count", &out->cfg.n_heads)) { if (err) *err = "no head_count"; return false; }
    if (gguf_get_u32(g, mk(a, "attention.head_count_kv"), &u)) out->cfg.n_kv_heads = (int)u; else out->cfg.n_kv_heads = out->cfg.n_heads;
    if (!need("expert_count", &out->mcfg.n_experts)) { if (err) *err = "no expert_count"; return false; }
    if (!need("expert_used_count", &out->mcfg.n_used)) { if (err) *err = "no expert_used_count"; return false; }
    if (!need("expert_feed_forward_length", &out->mcfg.expert_ffn) && !need("feed_forward_length", &out->mcfg.expert_ffn)) {
        if (err) *err = "no expert_ffn"; return false; }
    out->cfg.head_dim = out->cfg.dim / out->cfg.n_heads;
    out->cfg.ffn_dim = 0; out->cfg.eps = 1e-5f; out->cfg.rope_base = 10000.0f;
    if (gguf_get_f32(g, mk(a, "attention.layer_norm_rms_epsilon"), &f)) out->cfg.eps = f;
    if (gguf_get_f32(g, mk(a, "rope.freq_base"), &f)) out->cfg.rope_base = f;
    out->mcfg.has_shared = false;
    return true;
}

MoeLayerBlob layout(const LlamaConfig& c, const MoeConfig& mc) {
    size_t dim = c.dim, qd = (size_t)c.n_heads*c.head_dim, kvd = (size_t)c.n_kv_heads*c.head_dim, ef = mc.expert_ffn, E = mc.n_experts;
    MoeLayerBlob b;
    b.off_attn_norm = 0; b.off_wq = dim; b.off_wk = b.off_wq + dim*qd; b.off_wv = b.off_wk + dim*kvd;
    b.off_wo = b.off_wv + dim*kvd; b.off_ffn_norm = b.off_wo + qd*dim; b.off_router = b.off_ffn_norm + dim;
    b.off_experts = b.off_router + dim*E;
    b.off_egate = 0; b.off_eup = dim*ef; b.off_edown = 2*dim*ef; b.expert_stride = 3*dim*ef;
    b.total_elems = b.off_experts + E*b.expert_stride;
    return b;
}

// Dequant a tensor to a fresh device fp16 buffer (for the small non-streamed weights).
__half* dev_dequant(const GgufFile& g, const TensorInfo* t) {
    size_t n = gguf_tensor_elements(*t);
    size_t be, bb; block_info(t->ggml_type, &be, &bb);
    if (be == 0) return nullptr;
    size_t qbytes = (n / be) * bb;
    uint8_t* d_q; cudaMalloc(&d_q, qbytes);
    cudaMemcpy(d_q, gguf_tensor_data(g, *t), qbytes, cudaMemcpyHostToDevice);
    __half* d_out; cudaMalloc(&d_out, n * sizeof(__half));
    dequant_to_fp16(d_q, d_out, t->ggml_type, n, 0);
    cudaDeviceSynchronize(); cudaFree(d_q);
    return d_out;
}

}  // namespace

bool is_moe_model(const GgufFile& g) {
    std::string a; if (!gguf_get_str(g, "general.architecture", &a)) return false;
    uint32_t u; return gguf_get_u32(g, a + ".expert_count", &u) && u > 1;
}

bool load_moe_model(const GgufFile& g, LoadedMoeModel* out, std::string* err) {
    if (!read_moe_config(g, out, err)) return false;
    out->g = &g;
    const LlamaConfig& cfg = out->cfg; const MoeConfig& mc = out->mcfg;
    int dim = cfg.dim, ef = mc.expert_ffn, E = mc.n_experts;
    out->blob = layout(cfg, mc);

    const TensorInfo* embd = gguf_find_tensor(g, "token_embd.weight");
    if (!embd) { if (err) *err = "no token_embd"; return false; }
    out->vocab = embd->dims.size() > 1 ? (int)embd->dims[1] : (int)(gguf_tensor_elements(*embd)/dim);

    out->layers.resize(out->n_layers);
    for (int L = 0; L < out->n_layers; ++L) {
        std::vector<MatRef>& ms = out->layers[L];
        char nm[96];
        auto find = [&](const char* fmt, int e = -1) -> const TensorInfo* {
            if (e < 0) snprintf(nm, sizeof(nm), fmt, L); else snprintf(nm, sizeof(nm), fmt, L, e);
            return gguf_find_tensor(g, nm); };
        auto add = [&](const TensorInfo* t, size_t chunk_elems, int o, int in, size_t arena_off, bool is_norm, int expert = -1) -> bool {
            if (!t) return false;
            size_t be, bb; block_info(t->ggml_type, &be, &bb);
            if (be == 0) { if (err) *err = std::string("unsupported quant in ") + t->name; return false; }
            MatRef mr; mr.type = t->ggml_type; mr.out = o; mr.in = in;
            mr.n_elems = (size_t)o * in; mr.quant_bytes = (mr.n_elems / be) * bb;
            mr.src = gguf_tensor_data(g, *t) + (chunk_elems / be) * bb;
            mr.arena_off = arena_off; mr.is_norm = is_norm; mr.expert = expert;
            if (mr.quant_bytes > out->max_quant_bytes) out->max_quant_bytes = mr.quant_bytes;
            if (mr.n_elems > out->max_elems) out->max_elems = mr.n_elems;
            ms.push_back(mr); return true; };

        const TensorInfo *an = find("blk.%d.attn_norm.weight"), *fn = find("blk.%d.ffn_norm.weight");
        const TensorInfo *wq = find("blk.%d.attn_q.weight"), *wk = find("blk.%d.attn_k.weight");
        const TensorInfo *wv = find("blk.%d.attn_v.weight"), *wo = find("blk.%d.attn_output.weight");
        const TensorInfo *ri = find("blk.%d.ffn_gate_inp.weight");
        int o, in;
        if (!add(an, 0, dim, 1, out->blob.off_attn_norm, true)) { if (err&&err->empty()) *err="no attn_norm"; return false; }
        if (!add(fn, 0, dim, 1, out->blob.off_ffn_norm, true)) { if (err&&err->empty()) *err="no ffn_norm"; return false; }
        weight_out_in(*wq,&o,&in); if(!add(wq,0,o,in,out->blob.off_wq,false)) return false;
        weight_out_in(*wk,&o,&in); if(!add(wk,0,o,in,out->blob.off_wk,false)) return false;
        weight_out_in(*wv,&o,&in); if(!add(wv,0,o,in,out->blob.off_wv,false)) return false;
        weight_out_in(*wo,&o,&in); if(!add(wo,0,o,in,out->blob.off_wo,false)) return false;
        weight_out_in(*ri,&o,&in); if(!add(ri,0,o,in,out->blob.off_router,false)) return false;

        const TensorInfo *ge = find("blk.%d.ffn_gate_exps.weight"), *ue = find("blk.%d.ffn_up_exps.weight"), *de = find("blk.%d.ffn_down_exps.weight");
        for (int e = 0; e < E; ++e) {
            size_t eb = out->blob.off_experts + (size_t)e*out->blob.expert_stride;
            const TensorInfo *gpe = find("blk.%d.ffn_gate.%d.weight", e), *upe = find("blk.%d.ffn_up.%d.weight", e), *dpe = find("blk.%d.ffn_down.%d.weight", e);
            if (gpe && upe && dpe) {
                weight_out_in(*gpe,&o,&in); if(!add(gpe,0,o,in,eb+out->blob.off_egate,false,e)) return false;
                weight_out_in(*upe,&o,&in); if(!add(upe,0,o,in,eb+out->blob.off_eup,false,e)) return false;
                weight_out_in(*dpe,&o,&in); if(!add(dpe,0,o,in,eb+out->blob.off_edown,false,e)) return false;
            } else if (ge && ue && de) {
                if(!add(ge,(size_t)e*dim*ef,ef,dim,eb+out->blob.off_egate,false,e)) return false;
                if(!add(ue,(size_t)e*dim*ef,ef,dim,eb+out->blob.off_eup,false,e)) return false;
                if(!add(de,(size_t)e*ef*dim,dim,ef,eb+out->blob.off_edown,false,e)) return false;
            } else { if (err) *err = "missing expert tensors"; return false; }
        }
    }

    out->rw.token_embd = dev_dequant(g, embd);
    const TensorInfo* on = gguf_find_tensor(g, "output_norm.weight");
    const TensorInfo* ow = gguf_find_tensor(g, "output.weight");
    if (!on) { if (err) *err = "no output_norm"; return false; }
    out->rw.final_norm = dev_dequant(g, on);
    out->rw.output = ow ? dev_dequant(g, ow) : out->rw.token_embd;
    if (!out->rw.token_embd || !out->rw.final_norm || !out->rw.output) { if (err) *err = "non-layer dequant failed"; return false; }
    return true;
}

void free_moe_model(LoadedMoeModel* m) {
    if (m->rw.token_embd) cudaFree((void*)m->rw.token_embd);
    if (m->rw.final_norm) cudaFree((void*)m->rw.final_norm);
    if (m->rw.output && m->rw.output != m->rw.token_embd) cudaFree((void*)m->rw.output);
    m->rw.token_embd = nullptr;
}

bool moe_generate(LoadedMoeModel& m, std::vector<int>& ids, int n_gen, double budget_gb, std::string* err) {
    const LlamaConfig& cfg = m.cfg; const MoeConfig& mc = m.mcfg;
    int dim = cfg.dim, qd = cfg.n_heads*cfg.head_dim, kvd = cfg.n_kv_heads*cfg.head_dim, ef = mc.expert_ffn, E = mc.n_experts;
    int prompt_len = (int)ids.size(), max_T = prompt_len + n_gen;
    size_t per = m.blob.total_elems;

    __half *hidden, *normed, *arena, *d_deq, *logits_d; uint8_t* d_qstage; int* positions;
    cudaMalloc(&hidden,(size_t)max_T*dim*2); cudaMalloc(&normed,(size_t)max_T*dim*2);
    cudaMalloc(&arena, per*2); cudaMalloc(&d_deq, m.max_elems*2); cudaMalloc(&d_qstage, m.max_quant_bytes);
    cudaMalloc(&logits_d,(size_t)m.vocab*2); cudaMalloc(&positions, max_T*4);
    LayerScratch s;
    cudaMalloc(&s.xn,(size_t)max_T*dim*2); cudaMalloc(&s.q,(size_t)max_T*qd*2); cudaMalloc(&s.k,(size_t)max_T*kvd*2);
    cudaMalloc(&s.v,(size_t)max_T*kvd*2); cudaMalloc(&s.att,(size_t)max_T*qd*2); cudaMalloc(&s.proj,(size_t)max_T*dim*2);
    cudaMalloc(&s.gate,(size_t)max_T*ef*2); cudaMalloc(&s.up,(size_t)max_T*ef*2);
    MoeScratch ms;
    cudaMalloc(&ms.logits,(size_t)max_T*E*2); cudaMalloc(&ms.route_w,(size_t)max_T*E*sizeof(float));
    cudaMalloc(&ms.gate,(size_t)max_T*ef*2); cudaMalloc(&ms.up,(size_t)max_T*ef*2);
    cudaMalloc(&ms.ye,(size_t)max_T*dim*2); cudaMalloc(&ms.moe_out,(size_t)max_T*dim*2);
    cudaHostAlloc((void**)&ms.h_route,(size_t)max_T*E*sizeof(float), cudaHostAllocDefault);
    KVCache kv; kv.n_layers=m.n_layers; kv.max_T=max_T; kv.kvd=kvd; kv.len=0;
    cudaMalloc(&kv.K,(size_t)m.n_layers*max_T*kvd*2); cudaMalloc(&kv.V,(size_t)m.n_layers*max_T*kvd*2);

    double arena_gb = per*2/1e9;
    int K = budget_gb > 0 ? (int)(budget_gb / arena_gb) : 1;
    if (K < 1) K = 1;
    printf("per-layer arena=%.2f GB, budget=%.1f GB -> up to K=%d layers fit (streaming 1 at a time)\n",
           arena_gb, budget_gb, K);

    std::vector<const __half*> wg(E), wu(E), wd(E);
    for (int e=0;e<E;++e){ size_t eb=m.blob.off_experts+(size_t)e*m.blob.expert_stride;
        wg[e]=arena+eb+m.blob.off_egate; wu[e]=arena+eb+m.blob.off_eup; wd[e]=arena+eb+m.blob.off_edown; }
    MoeLayerWeights w;
    w.attn_norm=arena+m.blob.off_attn_norm; w.wq=arena+m.blob.off_wq; w.wk=arena+m.blob.off_wk; w.wv=arena+m.blob.off_wv;
    w.wo=arena+m.blob.off_wo; w.ffn_norm=arena+m.blob.off_ffn_norm; w.w_gate=arena+m.blob.off_router;
    w.wgate=wg.data(); w.wup=wu.data(); w.wdown=wd.data();

    int* d_ids; cudaMalloc(&d_ids, max_T*4);
    // cm = compute stream, cs = copy stream. Expert weights prefetch on cs while cm
    // computes the previous expert; a per-expert event orders each compute after its copy.
    cudaStream_t cm, cs; cudaStreamCreate(&cm); cudaStreamCreate(&cs);
    std::vector<cudaEvent_t> ev(E);
    for (int e=0;e<E;++e) cudaEventCreateWithFlags(&ev[e], cudaEventDisableTiming);
    Gemm gemm; gemm_create(&gemm);
    std::vector<__half> hl(m.vocab); std::vector<float> lf(m.vocab);

    auto stream_mat = [&](const MatRef& mr, cudaStream_t st) {
        cudaMemcpyAsync(d_qstage, mr.src, mr.quant_bytes, cudaMemcpyHostToDevice, st);
        if (mr.is_norm) dequant_to_fp16(d_qstage, arena + mr.arena_off, mr.type, mr.n_elems, st);
        else { dequant_to_fp16(d_qstage, d_deq, mr.type, mr.n_elems, st);
               transpose_fp16(d_deq, arena + mr.arena_off, mr.out, mr.in, st); }
    };
    // Active-expert streaming (PLAN sec 16) + copy/compute overlap: stream attn + router,
    // route, then prefetch ONLY the selected experts on cs while cm computes them.
    long exp_streamed = 0, exp_total = 0;
    auto forward = [&](const int* d_tok, int n_new, int len_before) {
        std::vector<int> pos(n_new); for (int i=0;i<n_new;++i) pos[i]=len_before+i;
        cudaMemcpy(positions, pos.data(), n_new*4, cudaMemcpyHostToDevice);
        embed_tokens(m.rw.token_embd, d_tok, hidden, n_new, dim, cm);
        int active[256];
        for (int L=0; L<m.n_layers; ++L) {
            for (const MatRef& mr : m.layers[L]) if (mr.expert < 0) stream_mat(mr, cm);   // attn + router (on cm; router syncs)
            int na = moe_attn_route(cfg, mc, w, hidden, positions, n_new, L, len_before, kv, s, ms, &gemm, cm, active, 256);
            moe_mlp_zero(ms.moe_out, n_new*dim, cm);
            // Prefetch each active expert on cs (staging buffers are free: the router sync
            // drained cm's attn streaming). Each records an event when its arena slot is ready.
            for (int a=0; a<na; ++a) {
                for (const MatRef& mr : m.layers[L]) if (mr.expert == active[a]) stream_mat(mr, cs);
                cudaEventRecord(ev[a], cs);
            }
            // Compute expert a on cm after its copy completes; cs streams a+1.. concurrently.
            // Waiting ev[na-1] also hands the shared staging buffers back to cm for next layer.
            for (int a=0; a<na; ++a) {
                cudaStreamWaitEvent(cm, ev[a], 0);
                moe_mlp_one(&gemm, s.xn, wg[active[a]], wu[active[a]], wd[active[a]], ms.route_w,
                            ms.moe_out, ms.gate, ms.up, ms.ye, n_new, dim, ef, active[a], E, cm);
            }
            moe_experts_tail(cfg, mc, w, hidden, n_new, s, ms, &gemm, cm);
            exp_streamed += na; exp_total += E;
        }
        rmsnorm(hidden, m.rw.final_norm, normed, n_new, dim, cfg.eps, cm);
        gemm_rowmajor(&gemm, m.rw.output, normed+(size_t)(n_new-1)*dim, logits_d, m.vocab, 1, dim, cm);
        cudaStreamSynchronize(cm);
    };
    auto sample = [&]() -> int {
        cudaMemcpy(hl.data(), logits_d, m.vocab*2, cudaMemcpyDeviceToHost);
        for (int v=0;v<m.vocab;++v) lf[v]=__half2float(hl[v]);
        return sample_greedy(lf.data(), m.vocab);
    };

    printf("prefill %d + generate %d (MoE, stream-original)...\n", prompt_len, n_gen);
    cudaMemcpy(d_ids, ids.data(), prompt_len*4, cudaMemcpyHostToDevice);
    forward(d_ids, prompt_len, 0); kv.len = prompt_len;
    int next = sample(); ids.push_back(next);
    auto t0 = std::chrono::steady_clock::now();
    for (int step=1; step<n_gen; ++step) {
        cudaMemcpy(d_ids, &next, 4, cudaMemcpyHostToDevice);
        forward(d_ids, 1, kv.len); kv.len += 1; next = sample(); ids.push_back(next);
    }
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { if (err) *err = cudaGetErrorString(e); return false; }
    if (n_gen > 1) printf("%.2f tokens/s decode\n", (n_gen-1)/secs);
    if (exp_total > 0)
        printf("active-expert streaming: %ld / %ld expert-loads (%.1f%% of dense), %d/%d used/layer\n",
               exp_streamed, exp_total, 100.0 * exp_streamed / exp_total, mc.n_used, E);

    for (int e=0;e<E;++e) cudaEventDestroy(ev[e]);
    cudaStreamDestroy(cs);
    cudaFree(kv.K); cudaFree(kv.V); gemm_destroy(&gemm); free_moe_model(&m);
    return true;
}

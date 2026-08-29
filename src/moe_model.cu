// MoE model: stream original quantized weights, dequant + transpose on the GPU.
#include "moe_model.h"
#include "weights_util.h"
#include "dequant.h"
#include "rmsnorm.h"
#include "runner.h"
#include "sampling.h"
#include "gemm.h"
#include <cstdio>
#include <cstdlib>
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

// SentencePiece detokenize of one token piece: U+2581 (▁) -> space, <0xAB> -> raw byte.
std::string sp_piece(const std::string& t) {
    if (t.size() == 6 && t[0] == '<' && t[1] == '0' && t[2] == 'x' && t[5] == '>')
        return std::string(1, (char)strtol(t.c_str() + 3, nullptr, 16));
    std::string o; o.reserve(t.size());
    for (size_t i = 0; i < t.size();) {
        if ((unsigned char)t[i] == 0xE2 && i + 2 < t.size() &&
            (unsigned char)t[i+1] == 0x96 && (unsigned char)t[i+2] == 0x81) { o.push_back(' '); i += 3; }
        else { o.push_back(t[i]); ++i; }
    }
    return o;
}

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

// Stream one weight matrix (DMA quant -> GPU dequant -> transpose into the arena) on `st`.
static void session_stream_mat(MoeSession& S, const MatRef& mr, cudaStream_t st) {
    cudaMemcpyAsync(S.d_qstage, mr.src, mr.quant_bytes, cudaMemcpyHostToDevice, st);
    if (mr.is_norm) dequant_to_fp16(S.d_qstage, S.arena + mr.arena_off, mr.type, mr.n_elems, st);
    else { dequant_to_fp16(S.d_qstage, S.d_deq, mr.type, mr.n_elems, st);
           transpose_fp16(S.d_deq, S.arena + mr.arena_off, mr.out, mr.in, st); }
}

bool moe_session_init(LoadedMoeModel& m, int max_T, double budget_gb, MoeSession* S, std::string* err) {
    const LlamaConfig& cfg = m.cfg; const MoeConfig& mc = m.mcfg;
    int dim = cfg.dim, qd = cfg.n_heads*cfg.head_dim, kvd = cfg.n_kv_heads*cfg.head_dim, ef = mc.expert_ffn, E = mc.n_experts;
    size_t per = m.blob.total_elems;
    S->m = &m; S->max_T = max_T; S->budget_gb = budget_gb; S->len = 0;

    cudaMalloc(&S->hidden,(size_t)max_T*dim*2); cudaMalloc(&S->normed,(size_t)max_T*dim*2);
    cudaMalloc(&S->arena, per*2); cudaMalloc(&S->d_deq, m.max_elems*2); cudaMalloc(&S->d_qstage, m.max_quant_bytes);
    cudaMalloc(&S->logits_d,(size_t)m.vocab*2); cudaMalloc(&S->positions, max_T*4);
    cudaMalloc(&S->s.xn,(size_t)max_T*dim*2); cudaMalloc(&S->s.q,(size_t)max_T*qd*2); cudaMalloc(&S->s.k,(size_t)max_T*kvd*2);
    cudaMalloc(&S->s.v,(size_t)max_T*kvd*2); cudaMalloc(&S->s.att,(size_t)max_T*qd*2); cudaMalloc(&S->s.proj,(size_t)max_T*dim*2);
    cudaMalloc(&S->s.gate,(size_t)max_T*ef*2); cudaMalloc(&S->s.up,(size_t)max_T*ef*2);
    cudaMalloc(&S->ms.logits,(size_t)max_T*E*2); cudaMalloc(&S->ms.route_w,(size_t)max_T*E*sizeof(float));
    cudaMalloc(&S->ms.gate,(size_t)max_T*ef*2); cudaMalloc(&S->ms.up,(size_t)max_T*ef*2);
    cudaMalloc(&S->ms.ye,(size_t)max_T*dim*2); cudaMalloc(&S->ms.moe_out,(size_t)max_T*dim*2);
    cudaHostAlloc((void**)&S->ms.h_route,(size_t)max_T*E*sizeof(float), cudaHostAllocDefault);
    S->kv.n_layers=m.n_layers; S->kv.max_T=max_T; S->kv.kvd=kvd; S->kv.len=0;
    cudaMalloc(&S->kv.K,(size_t)m.n_layers*max_T*kvd*2); cudaMalloc(&S->kv.V,(size_t)m.n_layers*max_T*kvd*2);

    double arena_gb = per*2/1e9;
    int K = budget_gb > 0 ? (int)(budget_gb / arena_gb) : 1; if (K < 1) K = 1;
    printf("per-layer arena=%.2f GB, budget=%.1f GB -> up to K=%d layers fit (streaming 1 at a time)\n",
           arena_gb, budget_gb, K);

    S->wg.resize(E); S->wu.resize(E); S->wd.resize(E);
    for (int e=0;e<E;++e){ size_t eb=m.blob.off_experts+(size_t)e*m.blob.expert_stride;
        S->wg[e]=S->arena+eb+m.blob.off_egate; S->wu[e]=S->arena+eb+m.blob.off_eup; S->wd[e]=S->arena+eb+m.blob.off_edown; }
    S->w.attn_norm=S->arena+m.blob.off_attn_norm; S->w.wq=S->arena+m.blob.off_wq; S->w.wk=S->arena+m.blob.off_wk;
    S->w.wv=S->arena+m.blob.off_wv; S->w.wo=S->arena+m.blob.off_wo; S->w.ffn_norm=S->arena+m.blob.off_ffn_norm;
    S->w.w_gate=S->arena+m.blob.off_router; S->w.wgate=S->wg.data(); S->w.wup=S->wu.data(); S->w.wdown=S->wd.data();

    cudaMalloc(&S->d_ids, max_T*4);
    cudaStreamCreate(&S->cm); cudaStreamCreate(&S->cs);
    S->ev.resize(E); for (int e=0;e<E;++e) cudaEventCreateWithFlags(&S->ev[e], cudaEventDisableTiming);
    gemm_create(&S->gemm);
    S->hl.resize(m.vocab); S->lf.resize(m.vocab);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { if (err) *err = std::string("session_init: ") + cudaGetErrorString(e); return false; }
    return true;
}

bool moe_session_eval(MoeSession& S, const int* ids, int n_new, std::string* err) {
    LoadedMoeModel& m = *S.m; const LlamaConfig& cfg = m.cfg; const MoeConfig& mc = m.mcfg;
    int dim = cfg.dim, ef = mc.expert_ffn, E = mc.n_experts, len_before = S.len;
    if (S.len + n_new > S.max_T) { if (err) *err = "context length exceeded (raise --ctx)"; return false; }

    cudaMemcpy(S.d_ids, ids, n_new*4, cudaMemcpyHostToDevice);
    std::vector<int> pos(n_new); for (int i=0;i<n_new;++i) pos[i]=len_before+i;
    cudaMemcpy(S.positions, pos.data(), n_new*4, cudaMemcpyHostToDevice);
    embed_tokens(m.rw.token_embd, S.d_ids, S.hidden, n_new, dim, S.cm);
    int active[256];
    for (int L=0; L<m.n_layers; ++L) {
        for (const MatRef& mr : m.layers[L]) if (mr.expert < 0) session_stream_mat(S, mr, S.cm);   // attn + router
        int na = moe_attn_route(cfg, mc, S.w, S.hidden, S.positions, n_new, L, len_before, S.kv, S.s, S.ms, &S.gemm, S.cm, active, 256);
        moe_mlp_zero(S.ms.moe_out, n_new*dim, S.cm);
        for (int a=0; a<na; ++a) {                             // prefetch active experts on cs
            for (const MatRef& mr : m.layers[L]) if (mr.expert == active[a]) session_stream_mat(S, mr, S.cs);
            cudaEventRecord(S.ev[a], S.cs);
        }
        for (int a=0; a<na; ++a) {                             // compute on cm, overlapped with cs
            cudaStreamWaitEvent(S.cm, S.ev[a], 0);
            moe_mlp_one(&S.gemm, S.s.xn, S.wg[active[a]], S.wu[active[a]], S.wd[active[a]], S.ms.route_w,
                        S.ms.moe_out, S.ms.gate, S.ms.up, S.ms.ye, n_new, dim, ef, active[a], E, S.cm);
        }
        moe_experts_tail(cfg, mc, S.w, S.hidden, n_new, S.s, S.ms, &S.gemm, S.cm);
        S.exp_streamed += na; S.exp_total += E;
    }
    rmsnorm(S.hidden, m.rw.final_norm, S.normed, n_new, dim, cfg.eps, S.cm);
    gemm_rowmajor(&S.gemm, m.rw.output, S.normed+(size_t)(n_new-1)*dim, S.logits_d, m.vocab, 1, dim, S.cm);
    cudaStreamSynchronize(S.cm);
    S.len += n_new; S.kv.len = S.len;
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { if (err) *err = cudaGetErrorString(e); return false; }
    return true;
}

int moe_session_sample(MoeSession& S, float temp, float rand01) {
    int vocab = S.m->vocab;
    cudaMemcpy(S.hl.data(), S.logits_d, vocab*2, cudaMemcpyDeviceToHost);
    for (int v=0;v<vocab;++v) S.lf[v]=__half2float(S.hl[v]);
    return temp > 0.0f ? sample_temperature(S.lf.data(), vocab, temp, rand01)
                       : sample_greedy(S.lf.data(), vocab);
}

void moe_session_reset(MoeSession& S) { S.len = 0; S.kv.len = 0; }

void moe_session_free(MoeSession* S) {
    if (!S->m) return;
    for (auto ev : S->ev) cudaEventDestroy(ev);
    if (S->cs) cudaStreamDestroy(S->cs); if (S->cm) cudaStreamDestroy(S->cm);
    gemm_destroy(&S->gemm);
    cudaFree(S->hidden); cudaFree(S->normed); cudaFree(S->arena); cudaFree(S->d_deq); cudaFree(S->d_qstage);
    cudaFree(S->logits_d); cudaFree(S->positions); cudaFree(S->d_ids);
    cudaFree(S->s.xn); cudaFree(S->s.q); cudaFree(S->s.k); cudaFree(S->s.v); cudaFree(S->s.att); cudaFree(S->s.proj);
    cudaFree(S->s.gate); cudaFree(S->s.up);
    cudaFree(S->ms.logits); cudaFree(S->ms.route_w); cudaFree(S->ms.gate); cudaFree(S->ms.up); cudaFree(S->ms.ye); cudaFree(S->ms.moe_out);
    cudaFreeHost(S->ms.h_route);
    cudaFree(S->kv.K); cudaFree(S->kv.V);
    S->m = nullptr;
}

bool moe_generate(LoadedMoeModel& m, std::vector<int>& ids, int n_gen, double budget_gb, std::string* err) {
    int prompt_len = (int)ids.size();
    MoeSession S;
    if (!moe_session_init(m, prompt_len + n_gen, budget_gb, &S, err)) return false;

    // Live token streaming (SEMILLM_STREAM=1) via the GGUF piece table.
    const std::vector<std::string>* toks = nullptr;
    auto tit = m.g->meta.find("tokenizer.ggml.tokens");
    if (tit != m.g->meta.end() && !tit->second.strs.empty()) toks = &tit->second.strs;
    bool stream_out = getenv("SEMILLM_STREAM") != nullptr;
    auto emit = [&](int id) {
        if (!stream_out || !toks || id < 0 || id >= (int)toks->size()) return;
        std::string p = sp_piece((*toks)[id]); fputs(p.c_str(), stdout); fflush(stdout);
    };

    printf("prefill %d + generate %d (MoE, stream-original)...\n", prompt_len, n_gen);
    if (stream_out) { fputs("> ", stdout); for (int id : ids) emit(id); }
    if (!moe_session_eval(S, ids.data(), prompt_len, err)) { moe_session_free(&S); return false; }
    int next = moe_session_sample(S, 0, 0); ids.push_back(next); emit(next);
    auto t0 = std::chrono::steady_clock::now();
    for (int step=1; step<n_gen; ++step) {
        if (!moe_session_eval(S, &next, 1, err)) { moe_session_free(&S); return false; }
        next = moe_session_sample(S, 0, 0); ids.push_back(next); emit(next);
    }
    if (stream_out) fputc('\n', stdout);
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
    if (n_gen > 1) printf("%.2f tokens/s decode\n", (n_gen-1)/secs);
    if (S.exp_total > 0)
        printf("active-expert streaming: %ld / %ld expert-loads (%.1f%% of dense), %d/%d used/layer\n",
               S.exp_streamed, S.exp_total, 100.0 * S.exp_streamed / S.exp_total, m.mcfg.n_used, m.mcfg.n_experts);

    moe_session_free(&S); free_moe_model(&m);
    return true;
}

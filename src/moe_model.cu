// MoE model: stream original quantized weights, dequant + transpose on the GPU.
#include "moe_model.h"
#include "weights_util.h"
#include "dequant.h"
#include "rmsnorm.h"
#include "runner.h"
#include "sampling.h"
#include "gemm.h"
#include "fused_gemm.h"
#include "rope.h"
#include "attention.h"
#include "elementwise.h"
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
        case GGML_Q2_K: *be = 256; *bb = 84;  break;
        case GGML_Q3_K: *be = 256; *bb = 110; break;
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
    if (!need("expert_count", &out->mcfg.n_experts) || out->mcfg.n_experts < 1) out->mcfg.n_experts = 1;   // dense = 1 "expert"
    if (!need("expert_used_count", &out->mcfg.n_used) || out->mcfg.n_used < 1) out->mcfg.n_used = 1;
    if (!need("expert_feed_forward_length", &out->mcfg.expert_ffn) && !need("feed_forward_length", &out->mcfg.expert_ffn)) {
        if (err) *err = "no expert_ffn / feed_forward_length"; return false; }
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
        if (ri) { weight_out_in(*ri,&o,&in); if(!add(ri,0,o,in,out->blob.off_router,false)) return false; }   // dense has no router

        // Dense (E==1): a single ffn_gate/up/down.weight is expert 0. MoE: per-expert tensors.
        const TensorInfo *gs = find("blk.%d.ffn_gate.weight"), *us = find("blk.%d.ffn_up.weight"), *ds = find("blk.%d.ffn_down.weight");
        if (E == 1 && gs && us && ds) {
            weight_out_in(*gs,&o,&in); if(!add(gs,0,o,in,out->blob.off_experts+out->blob.off_egate,false,0)) return false;
            weight_out_in(*us,&o,&in); if(!add(us,0,o,in,out->blob.off_experts+out->blob.off_eup,false,0)) return false;
            weight_out_in(*ds,&o,&in); if(!add(ds,0,o,in,out->blob.off_experts+out->blob.off_edown,false,0)) return false;
        } else {
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

// Stream one weight matrix (DMA quant -> GPU dequant -> transpose) into `dest` on `st`.
static void session_stream_to(MoeSession& S, const MatRef& mr, __half* dest, cudaStream_t st) {
    cudaMemcpyAsync(S.d_qstage, mr.src, mr.quant_bytes, cudaMemcpyHostToDevice, st);
    if (mr.is_norm) dequant_to_fp16(S.d_qstage, dest, mr.type, mr.n_elems, st);
    else { dequant_to_fp16(S.d_qstage, S.d_deq, mr.type, mr.n_elems, st);
           transpose_fp16(S.d_deq, dest, mr.out, mr.in, st); }
}
// Non-expert weights live at fixed offsets in the (small) per-layer arena.
static void session_stream_mat(MoeSession& S, const MatRef& mr, cudaStream_t st) {
    session_stream_to(S, mr, S.arena + mr.arena_off, st);
}

// Resolve expert `e` of layer `L` to a cache slot, searching only THIS layer's region
// (slots [L*per_layer, +per_layer)). *hit if already resident. Miss => LRU-evict within
// the layer's own region, so other layers are never disturbed.
static int cache_slot(ExpertCache& c, int L, int e, bool* hit) {
    int lo, hi, key;
    if (c.global) { lo = 0; hi = c.capacity; key = L; }     // dense: global LRU keyed by layer
    else { lo = L * c.per_layer; hi = lo + c.per_layer; key = e; }
    for (int s = lo; s < hi; ++s)
        if (c.slot_key[s] == key) { c.slot_lru[s] = ++c.tick; *hit = true; ++c.hits; return s; }
    int victim = lo; uint64_t best = UINT64_MAX;
    for (int s = lo; s < hi; ++s) {
        if (c.slot_key[s] < 0) { victim = s; break; }       // prefer an empty slot
        if (c.slot_lru[s] < best) { best = c.slot_lru[s]; victim = s; }
    }
    c.slot_key[victim] = key; c.slot_lru[victim] = ++c.tick;
    *hit = false; ++c.misses;
    return victim;
}

bool moe_session_init(LoadedMoeModel& m, int max_T, double budget_gb, MoeSession* S, std::string* err) {
    const LlamaConfig& cfg = m.cfg; const MoeConfig& mc = m.mcfg;
    int dim = cfg.dim, qd = cfg.n_heads*cfg.head_dim, kvd = cfg.n_kv_heads*cfg.head_dim, ef = mc.expert_ffn, E = mc.n_experts;
    S->m = &m; S->max_T = max_T; S->budget_gb = budget_gb; S->len = 0;

    // Experts now live in the hot-expert cache, so the arena only holds the (small)
    // non-expert weights of one layer: attn_norm | Wq | Wk | Wv | Wo | ffn_norm | router.
    size_t arena_elems = m.blob.off_experts;
    cudaMalloc(&S->hidden,(size_t)max_T*dim*2); cudaMalloc(&S->normed,(size_t)max_T*dim*2);
    cudaMalloc(&S->arena, arena_elems*2); cudaMalloc(&S->d_deq, m.max_elems*2); cudaMalloc(&S->d_qstage, m.max_quant_bytes);
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

    // Hot-expert cache sized from the budget, PARTITIONED PER LAYER:
    // per_layer = clamp(budget / one-expert-bytes / n_layers, n_used, n_experts).
    size_t stride = m.blob.expert_stride;                 // fp16 elems per expert (gate|up|down)
    double expert_gb = stride * 2.0 / 1e9;
    ExpertCache& c = S->cache;
    c.n_experts = E; c.stride = stride;
    int C;
    if (E == 1) {
        // Dense: the fused-decode path reads QUANTIZED weights (device quant cache below),
        // not the fp16 expert cache. Keep the fp16 cache minimal (used only by prefill's
        // old dequant path); the budget goes to the device quant cache (qpool).
        C = 2;
        c.global = true; c.capacity = C; c.per_layer = 1;
    } else {
        int per_layer = budget_gb > 0 ? (int)(budget_gb / expert_gb / m.n_layers) : mc.n_used;
        if (per_layer < mc.n_used) per_layer = mc.n_used;
        if (per_layer > E) per_layer = E;
        C = m.n_layers * per_layer;
        c.capacity = C; c.per_layer = per_layer;
        printf("hot-expert cache: %d/%d experts per layer resident (%.2f GB total) | budget=%.1f GB%s\n",
               per_layer, E, C*expert_gb, budget_gb,
               C*expert_gb > budget_gb + 0.01 ? "  (min n_used/layer exceeds budget)" : "");
    }
    cudaMalloc(&c.pool, (size_t)C * stride * 2);
    c.slot_key.assign(C, -1); c.slot_lru.assign(C, 0);
    c.slot_evt.resize(C); for (int i=0;i<C;++i) cudaEventCreateWithFlags(&c.slot_evt[i], cudaEventDisableTiming);

    // Device quant cache (dense fused-decode): copy as many whole layers' quantized weights
    // into device DRAM as the budget allows; the rest are read from the host mmap directly.
    S->qsrc.resize(m.n_layers);
    if (E == 1 && budget_gb > 0) {
        size_t budget_bytes = (size_t)(budget_gb * 1e9);
        std::vector<size_t> lb(m.n_layers, 0);
        for (int L = 0; L < m.n_layers; ++L) { size_t b = 0; for (auto& mr : m.layers[L]) b += mr.quant_bytes; lb[L] = b; }
        size_t acc = 0; int Kq = 0;
        for (int L = 0; L < m.n_layers; ++L) { if (acc + lb[L] <= budget_bytes) { acc += lb[L]; ++Kq; } else break; }
        if (Kq > 0) cudaMalloc(&S->qpool, acc);
        S->qres_layers = Kq;
        size_t off = 0;
        for (int L = 0; L < m.n_layers; ++L) {
            S->qsrc[L].resize(m.layers[L].size());
            for (size_t i = 0; i < m.layers[L].size(); ++i) {
                const MatRef& mr = m.layers[L][i];
                if (L < Kq) { cudaMemcpy(S->qpool + off, mr.src, mr.quant_bytes, cudaMemcpyHostToDevice);
                              S->qsrc[L][i] = S->qpool + off; off += mr.quant_bytes; }
                else S->qsrc[L][i] = mr.src;
            }
        }
        printf("dense fused decode: %d/%d layers resident in device quant cache (%.2f GB), rest from host mmap | budget=%.1f GB\n",
               Kq, m.n_layers, acc / 1e9, budget_gb);
    } else {
        for (int L = 0; L < m.n_layers; ++L) {
            S->qsrc[L].resize(m.layers[L].size());
            for (size_t i = 0; i < m.layers[L].size(); ++i) S->qsrc[L][i] = m.layers[L][i].src;
        }
    }

    S->wg.resize(E); S->wu.resize(E); S->wd.resize(E);   // per-layer active-expert ptrs (set in eval)
    S->w.attn_norm=S->arena+m.blob.off_attn_norm; S->w.wq=S->arena+m.blob.off_wq; S->w.wk=S->arena+m.blob.off_wk;
    S->w.wv=S->arena+m.blob.off_wv; S->w.wo=S->arena+m.blob.off_wo; S->w.ffn_norm=S->arena+m.blob.off_ffn_norm;
    S->w.w_gate=S->arena+m.blob.off_router; S->w.wgate=S->wg.data(); S->w.wup=S->wu.data(); S->w.wdown=S->wd.data();

    cudaMalloc(&S->d_ids, max_T*4); cudaMalloc(&S->d_arg, 4);
    cudaStreamCreate(&S->cm); cudaStreamCreate(&S->cs);
    S->ev.resize(E); for (int e=0;e<E;++e) cudaEventCreateWithFlags(&S->ev[e], cudaEventDisableTiming);
    gemm_create(&S->gemm);
    S->hl.resize(m.vocab); S->lf.resize(m.vocab);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { if (err) *err = std::string("session_init: ") + cudaGetErrorString(e); return false; }
    return true;
}

// True if every non-norm weight matrix in the model has a fused-GEMV kernel, so the
// dense-decode fast path can run without falling back mid-layer.
static bool dense_decode_supported(LoadedMoeModel& m) {
    if (m.mcfg.n_experts != 1) return false;
    for (auto& layer : m.layers)
        for (auto& mr : layer)
            if (!mr.is_norm) {
                uint32_t t = mr.type;
                if (t != 8 && t != 10 && t != 11 && t != 12 && t != 13 && t != 14) return false;  // Q8_0/Q2_K/Q3_K/Q4_K/Q5_K/Q6_K
            }
    return true;
}

// Fused dense layer for n_new tokens: compute straight from the mmap'd quantized weights
// via fused dequant-GEMV. No H2D staging, no fp16 dequant arena, no transpose, no expert
// cache -- GB10 unified memory lets the GEMV read the host mmap pointer (mr.src) directly
// (pageableMemoryAccess). Collapses the dequant + transpose cost. Used for decode (n_new=1)
// and short prefills (fused GEMV re-reads W per token, so only worthwhile for small n_new;
// long prompts keep the dequant-once + cuBLAS-GEMM path). Returns false on a structural
// mismatch.
static bool dense_decode_layer(MoeSession& S, int L, int len_before, int n_new, cudaStream_t st) {
    LoadedMoeModel& m = *S.m; const LlamaConfig& cfg = m.cfg;
    int dim = cfg.dim, qd = cfg.n_heads*cfg.head_dim, kvd = cfg.n_kv_heads*cfg.head_dim, ef = m.mcfg.expert_ffn;
    float scale = 1.0f / sqrtf((float)cfg.head_dim);
    const MoeLayerBlob& b = m.blob;
    // Resolve each role's MatRef + its effective src (device quant cache if resident, else
    // host mmap) by index into qsrc[L].
    struct Role { const MatRef* mr = nullptr; const uint8_t* src = nullptr; };
    Role an, fn, wq, wk, wv, wo, wg, wu, wd;
    const std::vector<MatRef>& layer = m.layers[L];
    for (size_t i = 0; i < layer.size(); ++i) {
        const MatRef& mr = layer[i]; const uint8_t* src = S.qsrc[L][i]; size_t off = mr.arena_off;
        if (mr.is_norm && off == b.off_attn_norm) { an.mr = &mr; an.src = src; }
        else if (mr.is_norm && off == b.off_ffn_norm) { fn.mr = &mr; fn.src = src; }
        else if (off == b.off_wq) { wq.mr = &mr; wq.src = src; }
        else if (off == b.off_wk) { wk.mr = &mr; wk.src = src; }
        else if (off == b.off_wv) { wv.mr = &mr; wv.src = src; }
        else if (off == b.off_wo) { wo.mr = &mr; wo.src = src; }
        else if (off == b.off_experts + b.off_egate) { wg.mr = &mr; wg.src = src; }
        else if (off == b.off_experts + b.off_eup) { wu.mr = &mr; wu.src = src; }
        else if (off == b.off_experts + b.off_edown) { wd.mr = &mr; wd.src = src; }
    }
    if (!an.mr||!fn.mr||!wq.mr||!wk.mr||!wv.mr||!wo.mr||!wg.mr||!wu.mr||!wd.mr) return false;

    // Norms are F32/F16 -> dequant into the (small) arena slots the old path used.
    __half* an_w = S.arena + b.off_attn_norm;
    __half* fn_w = S.arena + b.off_ffn_norm;
    dequant_to_fp16(an.src, an_w, an.mr->type, dim, st);
    dequant_to_fp16(fn.src, fn_w, fn.mr->type, dim, st);

    __half* Kbase = S.kv.K + (size_t)L * S.kv.max_T * kvd;
    __half* Vbase = S.kv.V + (size_t)L * S.kv.max_T * kvd;
    __half* Kdst = Kbase + (size_t)len_before * kvd;
    __half* Vdst = Vbase + (size_t)len_before * kvd;

    rmsnorm(S.hidden, an_w, S.s.xn, n_new, dim, cfg.eps, st);
    fused_gemv(wq.src, S.s.xn, S.s.q, n_new, qd, dim, wq.mr->type, st);
    fused_gemv(wk.src, S.s.xn, Kdst, n_new, kvd, dim, wk.mr->type, st);
    fused_gemv(wv.src, S.s.xn, Vdst, n_new, kvd, dim, wv.mr->type, st);
    rope_inplace(S.s.q, S.positions, n_new, cfg.n_heads, cfg.head_dim, cfg.rope_base, st);
    rope_inplace(Kdst, S.positions, n_new, cfg.n_kv_heads, cfg.head_dim, cfg.rope_base, st);
    attention_cached(S.s.q, Kbase, Vbase, S.s.att, n_new, len_before,
                     cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, scale, st);
    fused_gemv(wo.src, S.s.att, S.s.proj, n_new, dim, qd, wo.mr->type, st);
    residual_add(S.hidden, S.s.proj, n_new * dim, st);

    rmsnorm(S.hidden, fn_w, S.s.xn, n_new, dim, cfg.eps, st);
    fused_gemv(wg.src, S.s.xn, S.ms.gate, n_new, ef, dim, wg.mr->type, st);
    fused_gemv(wu.src, S.s.xn, S.ms.up, n_new, ef, dim, wu.mr->type, st);
    silu(S.ms.gate, S.ms.gate, n_new * ef, st);
    elementwise_mul(S.ms.gate, S.ms.up, S.ms.gate, n_new * ef, st);
    fused_gemv(wd.src, S.ms.gate, S.ms.ye, n_new, dim, ef, wd.mr->type, st);
    residual_add(S.hidden, S.ms.ye, n_new * dim, st);
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
    ExpertCache& c = S.cache;
    // Dense fused path: one-expert model, all weights fused-GEMV capable -> compute each
    // layer straight from mmap quant (no dequant arena/transpose/cache). Used for decode
    // and prefills up to a crossover: the fused GEMV re-reads W per token (~cost n_new x
    // one decode), while the old dequant+transpose+cuBLAS pass is a fixed ~1 model pass.
    // Measured crossover on 70B q4km is ~170 tokens; use 160 so typical prompts get the
    // fused path (avoids the ~55s naive-transpose prefill), long prompts keep GEMM.
    bool dense_fused = (E == 1 && n_new <= 160 && dense_decode_supported(m));
    int active[256]; bool miss[256]; int slot_of[256];
    for (int L=0; L<m.n_layers; ++L) {
        if (dense_fused) { dense_decode_layer(S, L, len_before, n_new, S.cm); continue; }
        for (const MatRef& mr : m.layers[L]) if (mr.expert < 0) session_stream_mat(S, mr, S.cm);   // attn + router
        int na = moe_attn_route(cfg, mc, S.w, S.hidden, S.positions, n_new, L, len_before, S.kv, S.s, S.ms, &S.gemm, S.cm, active, 256);
        moe_mlp_zero(S.ms.moe_out, n_new*dim, S.cm);
        // Process active experts in WAVES of per_layer: each wave's experts get distinct
        // cache slots (needed since the whole wave is resident during its compute), so na
        // may exceed per_layer (prefill routes many) without corrupting slots. Decode
        // (na <= n_used <= per_layer) is a single wave. Stream misses on cs, compute on cm.
        for (int w0 = 0; w0 < na; w0 += c.per_layer) {
            int wn = (na - w0 < c.per_layer) ? na - w0 : c.per_layer;
            for (int j = 0; j < wn; ++j) {
                int a = w0 + j, e = active[a]; bool hit;
                int slot = cache_slot(c, L, e, &hit); slot_of[a] = slot; miss[a] = !hit;
                __half* base = c.pool + (size_t)slot * c.stride;
                S.wg[e] = base; S.wu[e] = base + m.blob.off_eup; S.wd[e] = base + m.blob.off_edown;
                if (!hit) {
                    cudaStreamWaitEvent(S.cs, c.slot_evt[slot], 0);   // prior reader of this slot done
                    size_t eb = m.blob.off_experts + (size_t)e * m.blob.expert_stride;
                    for (const MatRef& mr : m.layers[L]) if (mr.expert == e)
                        session_stream_to(S, mr, base + (mr.arena_off - eb), S.cs);
                    cudaEventRecord(S.ev[j], S.cs);                   // copy done
                    S.exp_streamed += 1;
                }
            }
            for (int j = 0; j < wn; ++j) {
                int a = w0 + j, e = active[a];
                if (miss[a]) cudaStreamWaitEvent(S.cm, S.ev[j], 0);
                moe_mlp_one(&S.gemm, S.s.xn, S.wg[e], S.wu[e], S.wd[e], S.ms.route_w,
                            S.ms.moe_out, S.ms.gate, S.ms.up, S.ms.ye, n_new, dim, ef, e, E, S.cm);
                cudaEventRecord(c.slot_evt[slot_of[a]], S.cm);        // this slot's read is done
            }
        }
        moe_experts_tail(cfg, mc, S.w, S.hidden, n_new, S.s, S.ms, &S.gemm, S.cm);
        S.exp_total += na;
    }
    rmsnorm(S.hidden, m.rw.final_norm, S.normed, n_new, dim, cfg.eps, S.cm);
    gemm_rowmajor(&S.gemm, m.rw.output, S.normed+(size_t)(n_new-1)*dim, S.logits_d, m.vocab, 1, dim, S.cm);
    cudaStreamSynchronize(S.cm);
    S.len += n_new; S.kv.len = S.len;
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { if (err) *err = cudaGetErrorString(e); return false; }
    return true;
}

// GPU greedy argmax: one block reduces the whole vocab, so the common temp=0 path
// avoids copying the full logits vector to the host + a CPU scan every token.
__global__ void argmax_kernel(const __half* logits, int n, int* out) {
    __shared__ float sval[256]; __shared__ int sidx[256];
    int t = threadIdx.x; float best = -1e30f; int bi = 0;
    for (int v = t; v < n; v += blockDim.x) { float f = __half2float(logits[v]); if (f > best) { best = f; bi = v; } }
    sval[t] = best; sidx[t] = bi; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s && sval[t + s] > sval[t]) { sval[t] = sval[t + s]; sidx[t] = sidx[t + s]; }
        __syncthreads();
    }
    if (t == 0) out[0] = sidx[0];
}

int moe_session_sample(MoeSession& S, float temp, float rand01) {
    int vocab = S.m->vocab;
    if (temp <= 0.0f) {                                   // greedy: argmax on GPU, copy one int
        argmax_kernel<<<1, 256, 0, S.cm>>>(S.logits_d, vocab, S.d_arg);
        int idx = 0; cudaMemcpy(&idx, S.d_arg, 4, cudaMemcpyDeviceToHost);
        return idx;
    }
    cudaMemcpy(S.hl.data(), S.logits_d, vocab*2, cudaMemcpyDeviceToHost);
    for (int v=0;v<vocab;++v) S.lf[v]=__half2float(S.hl[v]);
    return sample_temperature(S.lf.data(), vocab, temp, rand01);
}

void moe_session_reset(MoeSession& S) { S.len = 0; S.kv.len = 0; }

void moe_session_free(MoeSession* S) {
    if (!S->m) return;
    for (auto ev : S->ev) cudaEventDestroy(ev);
    for (auto ev : S->cache.slot_evt) cudaEventDestroy(ev);
    cudaFree(S->cache.pool);
    if (S->qpool) cudaFree(S->qpool);
    if (S->cs) cudaStreamDestroy(S->cs); if (S->cm) cudaStreamDestroy(S->cm);
    gemm_destroy(&S->gemm);
    cudaFree(S->hidden); cudaFree(S->normed); cudaFree(S->arena); cudaFree(S->d_deq); cudaFree(S->d_qstage);
    cudaFree(S->logits_d); cudaFree(S->positions); cudaFree(S->d_ids); cudaFree(S->d_arg);
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
    long uses = S.cache.hits + S.cache.misses;
    long dense = (long)m.n_layers * m.mcfg.n_experts * n_gen;   // all experts, every forward
    if (uses > 0)
        printf("hot-expert cache: %ld/%ld hits (%.1f%%), %ld experts streamed (%.1f%% of %ld dense loads)\n",
               S.cache.hits, uses, 100.0*S.cache.hits/uses, S.cache.misses, 100.0*S.cache.misses/dense, dense);

    moe_session_free(&S); free_moe_model(&m);
    return true;
}

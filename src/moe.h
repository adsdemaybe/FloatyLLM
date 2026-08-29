// Mixture-of-Experts MLP: router (top-k gating) + per-expert SwiGLU, weighted sum.
// Dense-attention MoE (Mixtral-style). Naive: computes all experts (correctness
// first); expert streaming / top-k gather is the optimization (PLAN sec 16).
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "gemm.h"
#include "layer.h"

// Router: logits = x @ w_gate [T, n_experts]; softmax; keep top-k; renormalize.
// route_w (device float [T*n_experts]) gets the per-expert weights (0 off top-k).
void moe_router(Gemm* g, const __half* x, const __half* w_gate, __half* logits_TE,
                float* route_w, int T, int dim, int n_experts, int topk,
                cudaStream_t stream);

// MoE MLP: out[T,dim] = sum_e route_w[:,e] * down_e(silu(gate_e(x)) * up_e(x)).
// Computes ONLY the active experts (active[0..n_active), the ones any token routed
// to) - so decode with top-k does k expert FFNs, not n_experts. wgate/wup/wdown:
// arrays of n_experts device pointers, each [in,out]. gate/up/ye: scratch.
void moe_mlp(Gemm* g, const __half* x, const __half* const* wgate,
             const __half* const* wup, const __half* const* wdown,
             const float* route_w, __half* out, __half* gate, __half* up, __half* ye,
             int T, int dim, int ffn, int n_experts, const int* active, int n_active,
             cudaStream_t stream);

// --- Full MoE decoder layer (attention + MoE MLP) ---
struct MoeConfig {
    int n_experts = 0, n_used = 0, expert_ffn = 0;
    bool has_shared = false;
};

struct MoeLayerWeights {
    const __half *attn_norm, *wq, *wk, *wv, *wo, *ffn_norm;
    const __half* w_gate;                 // router [dim, n_experts]
    const __half* const* wgate;           // [n_experts], each [dim, expert_ffn]
    const __half* const* wup;
    const __half* const* wdown;
    const __half *sh_gate = nullptr, *sh_up = nullptr, *sh_down = nullptr;  // shared expert
};

struct MoeScratch {
    __half* logits;   // [n_new * n_experts]
    float*  route_w;  // [n_new * n_experts]
    __half* gate;     // [n_new * expert_ffn]
    __half* up;       // [n_new * expert_ffn]
    __half* ye;       // [n_new * dim]
    __half* moe_out;  // [n_new * dim]
    float*  h_route;  // pinned host [n_new * n_experts] (to pick active experts)
};

// Split MoE layer for active-expert streaming (PLAN sec 16): run attention + router,
// return the active experts in active[0..ret) (any token routed to them, <= max_active).
// Leaves the ffn-normed expert input in s.xn for moe_experts_out. The caller streams
// ONLY these experts' weights, then calls moe_experts_out.
int moe_attn_route(const LlamaConfig& cfg, const MoeConfig& mcfg, const MoeLayerWeights& w,
                   __half* hidden, const int* d_positions, int n_new, int layer_idx, int len_before,
                   KVCache& kv, LayerScratch& s, MoeScratch& ms, Gemm* gemm, cudaStream_t stream,
                   int* active, int max_active);

// Expert FFN over active[0..n_active) + optional shared expert + residual into hidden.
// Reads s.xn (set by moe_attn_route); wgate/wup/wdown[e] must be valid for each active e.
void moe_experts_out(const LlamaConfig& cfg, const MoeConfig& mcfg, const MoeLayerWeights& w,
                     __half* hidden, int n_new, LayerScratch& s, MoeScratch& ms, Gemm* gemm,
                     cudaStream_t stream, const int* active, int n_active);

// Cached MoE decoder layer (all experts resident): attention (KV cache) + router + experts.
void moe_layer_forward_cached(const LlamaConfig& cfg, const MoeConfig& mcfg,
                              const MoeLayerWeights& w, __half* hidden, const int* d_positions,
                              int n_new, int layer_idx, int len_before, KVCache& kv,
                              LayerScratch& s, MoeScratch& ms, Gemm* gemm, cudaStream_t stream);

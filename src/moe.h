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
// wgate/wup/wdown: arrays of n_experts device pointers, each [in,out] row-major.
// gate/up/ye: scratch [T*ffn], [T*ffn], [T*dim].
void moe_mlp(Gemm* g, const __half* x, const __half* const* wgate,
             const __half* const* wup, const __half* const* wdown,
             const float* route_w, __half* out, __half* gate, __half* up, __half* ye,
             int T, int dim, int ffn, int n_experts, cudaStream_t stream);

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
};

// Cached MoE decoder layer: attention (KV cache) + router + experts (+ shared).
void moe_layer_forward_cached(const LlamaConfig& cfg, const MoeConfig& mcfg,
                              const MoeLayerWeights& w, __half* hidden, const int* d_positions,
                              int n_new, int layer_idx, int len_before, KVCache& kv,
                              LayerScratch& s, MoeScratch& ms, Gemm* gemm, cudaStream_t stream);

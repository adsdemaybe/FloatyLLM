// Mixture-of-Experts MLP: router (top-k gating) + per-expert SwiGLU, weighted sum.
// Dense-attention MoE (Mixtral-style). Naive: computes all experts (correctness
// first); expert streaming / top-k gather is the optimization (PLAN sec 16).
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "gemm.h"

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

// One Llama decoder layer: rmsnorm -> QKV -> RoPE -> attention -> O -> residual
// -> rmsnorm -> SwiGLU MLP -> residual. Weights row-major [in, out], fp16. PLAN sec 7.
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "gemm.h"

struct LlamaConfig {
    int dim, n_heads, n_kv_heads, head_dim, ffn_dim;
    float eps, rope_base;
};

struct LayerWeights {
    const __half *attn_norm, *wq, *wk, *wv, *wo;
    const __half *ffn_norm, *wgate, *wup, *wdown;
};

// Device scratch, caller-allocated. Sizes for n_tokens T:
//   xn[T*dim], q[T*n_heads*head_dim], k/v[T*n_kv_heads*head_dim],
//   att[T*n_heads*head_dim], proj[T*dim], gate/up[T*ffn_dim].
struct LayerScratch {
    __half *xn, *q, *k, *v, *att, *proj, *gate, *up;
};

// In-place on hidden[n_tokens*dim]. d_positions[n_tokens] on device.
void layer_forward(const LlamaConfig& cfg, const LayerWeights& w,
                   __half* hidden, const int* d_positions, int n_tokens,
                   LayerScratch& s, Gemm* gemm, cudaStream_t stream);

// Llama decoder layer assembly. Wires the individual kernels; all work is
// enqueued on one stream (data deps enforce ordering within it).
#include "layer.h"
#include "rmsnorm.h"
#include "rope.h"
#include "attention.h"
#include "elementwise.h"
#include <cmath>

void layer_forward(const LlamaConfig& cfg, const LayerWeights& w,
                   __half* hidden, const int* d_positions, int n_tokens,
                   LayerScratch& s, Gemm* gemm, cudaStream_t stream) {
    int T = n_tokens;
    int qd = cfg.n_heads * cfg.head_dim;
    int kvd = cfg.n_kv_heads * cfg.head_dim;
    float scale = 1.0f / sqrtf((float)cfg.head_dim);

    // Attention block.
    rmsnorm(hidden, w.attn_norm, s.xn, T, cfg.dim, cfg.eps, stream);
    gemm_rowmajor(gemm, s.xn, w.wq, s.q, T, qd, cfg.dim, stream);
    gemm_rowmajor(gemm, s.xn, w.wk, s.k, T, kvd, cfg.dim, stream);
    gemm_rowmajor(gemm, s.xn, w.wv, s.v, T, kvd, cfg.dim, stream);
    rope_inplace(s.q, d_positions, T, cfg.n_heads, cfg.head_dim, cfg.rope_base, stream);
    rope_inplace(s.k, d_positions, T, cfg.n_kv_heads, cfg.head_dim, cfg.rope_base, stream);
    attention(s.q, s.k, s.v, s.att, T, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, scale, stream);
    gemm_rowmajor(gemm, s.att, w.wo, s.proj, T, cfg.dim, qd, stream);
    residual_add(hidden, s.proj, T * cfg.dim, stream);

    // MLP block (SwiGLU).
    rmsnorm(hidden, w.ffn_norm, s.xn, T, cfg.dim, cfg.eps, stream);
    gemm_rowmajor(gemm, s.xn, w.wgate, s.gate, T, cfg.ffn_dim, cfg.dim, stream);
    gemm_rowmajor(gemm, s.xn, w.wup, s.up, T, cfg.ffn_dim, cfg.dim, stream);
    silu(s.gate, s.gate, T * cfg.ffn_dim, stream);
    elementwise_mul(s.gate, s.up, s.gate, T * cfg.ffn_dim, stream);
    gemm_rowmajor(gemm, s.gate, w.wdown, s.proj, T, cfg.dim, cfg.ffn_dim, stream);
    residual_add(hidden, s.proj, T * cfg.dim, stream);
}

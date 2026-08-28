// Naive causal multi-head / grouped-query attention (correctness-first).
// Online-softmax accumulation (flash-style), no O(seq) scratch. Swap for
// FlashAttention later for performance. head_dim must be <= 128.
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Q: [n_tokens][n_heads][head_dim], K/V: [n_tokens][n_kv_heads][head_dim].
// out: [n_tokens][n_heads][head_dim]. Causal: token i attends to j <= i.
// GQA: query head h maps to kv head h / (n_heads / n_kv_heads).
void attention(const __half* Q, const __half* K, const __half* V, __half* out,
               int n_tokens, int n_heads, int n_kv_heads, int head_dim,
               float scale, cudaStream_t stream);

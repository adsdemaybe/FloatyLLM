// Rotary position embedding (RoPE), interleaved pairs (2i, 2i+1) = original LLaMA style.
// NOTE: llama.cpp has NORM (interleaved) vs NEOX (half-split) variants; the target
// model's convention must be confirmed against llama.cpp before the golden match.
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// In-place RoPE on x laid out [n_tokens][n_heads][head_dim].
// positions[n_tokens] gives each token's absolute position. head_dim must be even.
void rope_inplace(__half* x, const int* positions, int n_tokens, int n_heads,
                  int head_dim, float base, cudaStream_t stream);

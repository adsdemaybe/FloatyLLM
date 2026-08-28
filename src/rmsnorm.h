// RMSNorm over the hidden dimension (Llama-style, no bias, no mean-subtraction).
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// For each of n_rows rows of length dim: out = x * rsqrt(mean(x^2) + eps) * weight.
// Reduction runs in fp32 for stability; io is fp16. Async on stream.
void rmsnorm(const __half* x, const __half* weight, __half* out,
             int n_rows, int dim, float eps, cudaStream_t stream);

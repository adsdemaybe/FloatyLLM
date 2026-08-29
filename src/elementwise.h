// Elementwise forward ops: SiLU, residual add, elementwise multiply (SwiGLU).
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// out[i] = x[i] * sigmoid(x[i]).
void silu(const __half* x, __half* out, int n, cudaStream_t stream);

// x[i] += y[i]  (residual connection, in-place on x).
void residual_add(__half* x, const __half* y, int n, cudaStream_t stream);

// out[i] = a[i] * b[i]  (e.g. SwiGLU: silu(gate) * up).
void elementwise_mul(const __half* a, const __half* b, __half* out, int n,
                     cudaStream_t stream);

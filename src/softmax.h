// Numerically-stable softmax over the last dimension.
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// For each of n_rows rows of length dim: out = softmax(x) (max-subtracted). Async.
void softmax(const __half* x, __half* out, int n_rows, int dim, cudaStream_t stream);

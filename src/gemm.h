// GEMM wrapper (cuBLAS, fp16 in/out, fp32 accumulate). Row-major convenience.
// Starts on cuBLAS for correctness; cuBLASLt (tuned + fused dequant) is a later swap.
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

struct Gemm {
    cublasHandle_t handle;
};

void gemm_create(Gemm* g);
void gemm_destroy(Gemm* g);

// Row-major: C[m,n] = A[m,k] * B[k,n]. All fp16, fp32 accumulate. Async on stream.
void gemm_rowmajor(Gemm* g, const __half* A, const __half* B, __half* C,
                   int m, int n, int k, cudaStream_t stream);

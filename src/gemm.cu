// GEMM wrapper. cuBLAS is column-major, so a row-major C = A*B is produced by
// the standard operand-swap: column-major C^T[n,m] = B[k,n]^col * A[m,k]^col.
#include "gemm.h"

void gemm_create(Gemm* g) {
    cublasCreate(&g->handle);
}

void gemm_destroy(Gemm* g) {
    cublasDestroy(g->handle);
}

void gemm_rowmajor(Gemm* g, const __half* A, const __half* B, __half* C,
                   int m, int n, int k, cudaStream_t stream) {
    if (m <= 0 || n <= 0 || k <= 0) return;
    cublasSetStream(g->handle, stream);
    float alpha = 1.0f, beta = 0.0f;
    // Compute column-major (n x m) = B(n x k) * A(k x m), which equals row-major C(m x n).
    cublasGemmEx(g->handle, CUBLAS_OP_N, CUBLAS_OP_N,
                 n, m, k,
                 &alpha,
                 B, CUDA_R_16F, n,
                 A, CUDA_R_16F, k,
                 &beta,
                 C, CUDA_R_16F, n,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

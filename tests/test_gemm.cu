// Unit test for the row-major GEMM wrapper. RUN on a GPU.
// Verifies C[m,n] = A[m,k] * B[k,n] vs a CPU reference (checks the operand-swap).
#include "gemm.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

int main() {
    const int m = 5, k = 7, n = 3;
    std::vector<__half> A(m * k), B(k * n), C(m * n);
    std::vector<float> Af(m * k), Bf(k * n), expected(m * n);
    for (int i = 0; i < m * k; ++i) { float v = 0.1f * (float)((i * 3) % 11 - 5); A[i] = __float2half(v); Af[i] = __half2float(A[i]); }
    for (int i = 0; i < k * n; ++i) { float v = 0.1f * (float)((i * 5) % 13 - 6); B[i] = __float2half(v); Bf[i] = __half2float(B[i]); }

    // Row-major reference: C[i,j] = sum_p A[i,p] * B[p,j].
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (int p = 0; p < k; ++p) acc += Af[i * k + p] * Bf[p * n + j];
            expected[i * n + j] = acc;
        }

    __half *dA, *dB, *dC;
    cudaMalloc(&dA, A.size() * sizeof(__half));
    cudaMalloc(&dB, B.size() * sizeof(__half));
    cudaMalloc(&dC, C.size() * sizeof(__half));
    cudaMemcpy(dA, A.data(), A.size() * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B.data(), B.size() * sizeof(__half), cudaMemcpyHostToDevice);

    Gemm g;
    gemm_create(&g);
    gemm_rowmajor(&g, dA, dB, dC, m, n, k, 0);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 2; }
    cudaMemcpy(C.data(), dC, C.size() * sizeof(__half), cudaMemcpyDeviceToHost);
    gemm_destroy(&g);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);

    int fails = 0;
    for (int i = 0; i < m * n; ++i) {
        float got = __half2float(C[i]);
        float tol = 2e-2f * (fabsf(expected[i]) + 1.0f);
        if (fabsf(got - expected[i]) > tol) { if (fails < 10) printf("mismatch i=%d got=%f exp=%f\n", i, got, expected[i]); ++fails; }
    }
    if (fails) { printf("FAIL: %d/%d\n", fails, m * n); return 1; }
    printf("PASS: gemm %dx%d = %dx%d * %dx%d\n", m, n, m, k, k, n);
    return 0;
}

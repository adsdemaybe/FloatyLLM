// Unit test for RMSNorm. Compiled by CI; RUN on a GPU (non-zero on mismatch).
#include "rmsnorm.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

static bool cuda_ok(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        printf("CUDA error at %s: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

int main() {
    const int rows = 4;
    const int dim = 128;
    const float eps = 1e-6f;

    std::vector<__half> x(rows * dim), w(dim), out(rows * dim);
    std::vector<float> expected(rows * dim);

    for (int j = 0; j < dim; ++j) w[j] = __float2half(1.0f + 0.01f * (float)j);
    for (int r = 0; r < rows; ++r) {
        for (int j = 0; j < dim; ++j) {
            float v = 0.5f * (float)((r * 13 + j * 7) % 21 - 10);
            x[r * dim + j] = __float2half(v);
        }
    }

    // Host reference in fp32 using the fp16-rounded inputs.
    for (int r = 0; r < rows; ++r) {
        float ss = 0.0f;
        for (int j = 0; j < dim; ++j) {
            float v = __half2float(x[r * dim + j]);
            ss += v * v;
        }
        float inv = 1.0f / sqrtf(ss / (float)dim + eps);
        for (int j = 0; j < dim; ++j) {
            expected[r * dim + j] =
                __half2float(x[r * dim + j]) * inv * __half2float(w[j]);
        }
    }

    __half *dx, *dw, *dout;
    if (!cuda_ok(cudaMalloc(&dx, x.size() * sizeof(__half)), "malloc x")) return 2;
    if (!cuda_ok(cudaMalloc(&dw, w.size() * sizeof(__half)), "malloc w")) return 2;
    if (!cuda_ok(cudaMalloc(&dout, out.size() * sizeof(__half)), "malloc out")) return 2;
    cudaMemcpy(dx, x.data(), x.size() * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(dw, w.data(), w.size() * sizeof(__half), cudaMemcpyHostToDevice);

    rmsnorm(dx, dw, dout, rows, dim, eps, 0);
    if (!cuda_ok(cudaDeviceSynchronize(), "sync")) return 2;
    cudaMemcpy(out.data(), dout, out.size() * sizeof(__half), cudaMemcpyDeviceToHost);
    cudaFree(dx); cudaFree(dw); cudaFree(dout);

    int fails = 0;
    for (int i = 0; i < rows * dim; ++i) {
        float got = __half2float(out[i]);
        float exp = expected[i];
        float tol = 2e-2f * (fabsf(exp) + 1.0f);
        if (fabsf(got - exp) > tol) {
            if (fails < 10) printf("mismatch i=%d got=%f exp=%f\n", i, got, exp);
            ++fails;
        }
    }
    if (fails) { printf("FAIL: %d/%d\n", fails, rows * dim); return 1; }
    printf("PASS: rmsnorm %d rows x %d dim\n", rows, dim);
    return 0;
}

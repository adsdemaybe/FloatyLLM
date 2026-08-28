// Unit test for RoPE. Compiled by CI; RUN on a GPU (non-zero on mismatch).
#include "rope.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

int main() {
    const int n_tokens = 3, n_heads = 2, head_dim = 8;
    const float base = 10000.0f;
    const int n = n_tokens * n_heads * head_dim;

    std::vector<__half> x(n);
    std::vector<float> xf(n);
    std::vector<int> pos = {0, 5, 11};
    for (int i = 0; i < n; ++i) {
        float v = 0.2f * (float)((i % 17) - 8);
        x[i] = __float2half(v);
        xf[i] = __half2float(x[i]);
    }

    // Host reference (fp32) using fp16-rounded inputs.
    int half = head_dim / 2;
    for (int blk = 0; blk < n_tokens * n_heads; ++blk) {
        int token = blk / n_heads;
        float* v = xf.data() + (size_t)blk * head_dim;
        for (int j = 0; j < half; ++j) {
            float freq = powf(base, -2.0f * (float)j / (float)head_dim);
            float ang = (float)pos[token] * freq;
            float c = cosf(ang), s = sinf(ang);
            float a = v[2 * j], b = v[2 * j + 1];
            v[2 * j]     = a * c - b * s;
            v[2 * j + 1] = a * s + b * c;
        }
    }

    __half* dx; int* dpos;
    cudaMalloc(&dx, n * sizeof(__half));
    cudaMalloc(&dpos, pos.size() * sizeof(int));
    cudaMemcpy(dx, x.data(), n * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(dpos, pos.data(), pos.size() * sizeof(int), cudaMemcpyHostToDevice);
    rope_inplace(dx, dpos, n_tokens, n_heads, head_dim, base, 0);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 2; }
    cudaMemcpy(x.data(), dx, n * sizeof(__half), cudaMemcpyDeviceToHost);
    cudaFree(dx); cudaFree(dpos);

    int fails = 0;
    for (int i = 0; i < n; ++i) {
        float got = __half2float(x[i]);
        float tol = 2e-2f * (fabsf(xf[i]) + 1.0f);
        if (fabsf(got - xf[i]) > tol) { if (fails < 10) printf("mismatch i=%d got=%f exp=%f\n", i, got, xf[i]); ++fails; }
    }
    if (fails) { printf("FAIL: %d/%d\n", fails, n); return 1; }
    printf("PASS: rope %d tokens x %d heads x %d dim\n", n_tokens, n_heads, head_dim);
    return 0;
}

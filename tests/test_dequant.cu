// Unit test for Q8_0 dequant: synthesize blocks, dequant on GPU, compare vs host reference.
// Compiled by CI; must be RUN on a GPU (returns non-zero on mismatch or CUDA error).
#include "dequant.h"
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
    const int n_blocks = 5;
    const int n = n_blocks * QK8_0;

    std::vector<BlockQ80> blocks(n_blocks);
    std::vector<float> expected(n);
    for (int b = 0; b < n_blocks; ++b) {
        float scale = 0.01f * (float)(b + 1);
        __half d = __float2half(scale);
        blocks[b].d = d;
        for (int j = 0; j < QK8_0; ++j) {
            int8_t q = (int8_t)((b * 7 + j * 3) % 251 - 125);
            blocks[b].qs[j] = q;
            // Reference uses the fp16-rounded scale, matching the kernel's inputs.
            expected[b * QK8_0 + j] = __half2float(d) * (float)q;
        }
    }

    BlockQ80* d_blocks = nullptr;
    __half* d_out = nullptr;
    if (!cuda_ok(cudaMalloc(&d_blocks, n_blocks * sizeof(BlockQ80)), "malloc blocks")) return 2;
    if (!cuda_ok(cudaMalloc(&d_out, n * sizeof(__half)), "malloc out")) return 2;
    if (!cuda_ok(cudaMemcpy(d_blocks, blocks.data(), n_blocks * sizeof(BlockQ80),
                            cudaMemcpyHostToDevice), "H2D")) return 2;

    dequant_q8_0(d_blocks, d_out, n_blocks, 0);
    if (!cuda_ok(cudaDeviceSynchronize(), "sync")) return 2;

    std::vector<__half> out(n);
    if (!cuda_ok(cudaMemcpy(out.data(), d_out, n * sizeof(__half),
                            cudaMemcpyDeviceToHost), "D2H")) return 2;
    cudaFree(d_blocks);
    cudaFree(d_out);

    int fails = 0;
    for (int i = 0; i < n; ++i) {
        float got = __half2float(out[i]);
        float exp = expected[i];
        float tol = 1e-2f * (fabsf(exp) + 1.0f);
        if (fabsf(got - exp) > tol) {
            if (fails < 10) printf("mismatch i=%d got=%f exp=%f\n", i, got, exp);
            ++fails;
        }
    }
    if (fails) {
        printf("FAIL: %d/%d mismatches\n", fails, n);
        return 1;
    }
    printf("PASS: %d values dequantized correctly\n", n);
    return 0;
}

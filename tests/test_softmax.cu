// Unit test for softmax. RUN on a GPU (non-zero on mismatch).
#include "softmax.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

int main() {
    const int rows = 3, dim = 100;
    std::vector<__half> x(rows * dim), out(rows * dim);
    std::vector<float> expected(rows * dim);
    for (int r = 0; r < rows; ++r)
        for (int j = 0; j < dim; ++j)
            x[r * dim + j] = __float2half(0.1f * (float)((r * 7 + j * 3) % 41 - 20));

    for (int r = 0; r < rows; ++r) {
        float m = -1e30f;
        for (int j = 0; j < dim; ++j) m = fmaxf(m, __half2float(x[r * dim + j]));
        float s = 0.0f;
        for (int j = 0; j < dim; ++j) { float e = expf(__half2float(x[r*dim+j]) - m); expected[r*dim+j] = e; s += e; }
        for (int j = 0; j < dim; ++j) expected[r * dim + j] /= s;
    }

    __half *dx, *dout;
    cudaMalloc(&dx, x.size() * sizeof(__half));
    cudaMalloc(&dout, out.size() * sizeof(__half));
    cudaMemcpy(dx, x.data(), x.size() * sizeof(__half), cudaMemcpyHostToDevice);
    softmax(dx, dout, rows, dim, 0);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 2; }
    cudaMemcpy(out.data(), dout, out.size() * sizeof(__half), cudaMemcpyDeviceToHost);
    cudaFree(dx); cudaFree(dout);

    int fails = 0;
    for (int i = 0; i < rows * dim; ++i) {
        float got = __half2float(out[i]);
        if (fabsf(got - expected[i]) > 3e-3f) { if (fails < 10) printf("mismatch i=%d got=%f exp=%f\n", i, got, expected[i]); ++fails; }
    }
    if (fails) { printf("FAIL: %d/%d\n", fails, rows * dim); return 1; }
    printf("PASS: softmax %d rows x %d dim\n", rows, dim);
    return 0;
}

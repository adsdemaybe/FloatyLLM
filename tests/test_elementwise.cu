// Unit test for elementwise ops (silu, residual add, mul). RUN on a GPU.
#include "elementwise.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

static float silu_ref(float v) { return v / (1.0f + expf(-v)); }

int main() {
    const int n = 257; // non-round size to exercise grid-stride
    std::vector<__half> a(n), b(n), out(n);
    for (int i = 0; i < n; ++i) {
        a[i] = __float2half(0.1f * (float)((i % 41) - 20));
        b[i] = __float2half(0.05f * (float)((i % 13) - 6));
    }

    __half *da, *db, *dout;
    cudaMalloc(&da, n * sizeof(__half));
    cudaMalloc(&db, n * sizeof(__half));
    cudaMalloc(&dout, n * sizeof(__half));
    cudaMemcpy(da, a.data(), n * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b.data(), n * sizeof(__half), cudaMemcpyHostToDevice);

    int fails = 0;

    // silu
    silu(da, dout, n, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(out.data(), dout, n * sizeof(__half), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i) {
        float exp = silu_ref(__half2float(a[i]));
        if (fabsf(__half2float(out[i]) - exp) > 2e-2f * (fabsf(exp) + 1.0f)) ++fails;
    }

    // mul
    elementwise_mul(da, db, dout, n, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(out.data(), dout, n * sizeof(__half), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i) {
        float exp = __half2float(a[i]) * __half2float(b[i]);
        if (fabsf(__half2float(out[i]) - exp) > 2e-2f * (fabsf(exp) + 1.0f)) ++fails;
    }

    // residual add (in-place on da's copy)
    __half* dx;
    cudaMalloc(&dx, n * sizeof(__half));
    cudaMemcpy(dx, a.data(), n * sizeof(__half), cudaMemcpyHostToDevice);
    residual_add(dx, db, n, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(out.data(), dx, n * sizeof(__half), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i) {
        float exp = __half2float(a[i]) + __half2float(b[i]);
        if (fabsf(__half2float(out[i]) - exp) > 2e-2f * (fabsf(exp) + 1.0f)) ++fails;
    }

    cudaFree(da); cudaFree(db); cudaFree(dout); cudaFree(dx);
    if (fails) { printf("FAIL: %d mismatches\n", fails); return 1; }
    printf("PASS: elementwise silu/mul/residual on %d elems\n", n);
    return 0;
}

// Fused dequant-GEMV correctness: compare the fused kernel (reads native quantized
// weights) against a CPU dequant + GEMV reference, for Q8_0 and Q4_K.
#include "fused_gemm.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_fp16.h>

static void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

static int run(const char* name, uint32_t type, int block_bytes, int block_elems,
               void (*fill_row_ref)(const uint8_t*, int, float*)) {
    int n_in = block_elems * 3, n_out = 17, m = 2;      // 3 blocks/row, odd n_out
    int nb = n_in / block_elems;
    std::vector<uint8_t> W((size_t)n_out * nb * block_bytes);
    for (auto& b : W) b = rand() & 0xFF;                 // random bytes (dequant formula applies to any)
    // Give Q8_0/Q4_K a sane fp16 scale so values don't blow up: overwrite the scale halfs.
    for (int o = 0; o < n_out; ++o) for (int b = 0; b < nb; ++b) {
        __half* d = (__half*)(W.data() + ((size_t)o * nb + b) * block_bytes);
        d[0] = __float2half(0.05f * (((o + b) % 7) + 1));
        if (type == 12) d[1] = __float2half(0.03f * ((b % 5) + 1));   // Q4_K dmin
    }
    std::vector<__half> x((size_t)m * n_in), y_gpu((size_t)m * n_out);
    std::vector<float> y_ref((size_t)m * n_out, 0.0f);
    for (auto& v : x) v = __float2half((rand() / (float)RAND_MAX) * 2.0f - 1.0f);

    // Reference: dequant each row into fp32, dot with x.
    std::vector<float> row(n_in);
    for (int o = 0; o < n_out; ++o) {
        for (int b = 0; b < nb; ++b) fill_row_ref(W.data() + ((size_t)o * nb + b) * block_bytes, b, row.data());
        for (int t = 0; t < m; ++t) { float s = 0; for (int i = 0; i < n_in; ++i) s += row[i] * __half2float(x[(size_t)t * n_in + i]); y_ref[(size_t)t * n_out + o] = s; }
    }

    uint8_t* dW; __half *dx, *dy;
    cudaMalloc(&dW, W.size()); cudaMalloc(&dx, x.size() * 2); cudaMalloc(&dy, y_gpu.size() * 2);
    cudaMemcpy(dW, W.data(), W.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice);
    if (!fused_gemv(dW, dx, dy, m, n_out, n_in, type, 0)) { printf("FAIL %s: no kernel\n", name); return 1; }
    cudaDeviceSynchronize();
    cudaMemcpy(y_gpu.data(), dy, y_gpu.size() * 2, cudaMemcpyDeviceToHost);
    cudaFree(dW); cudaFree(dx); cudaFree(dy);

    float max_err = 0;
    for (size_t i = 0; i < y_ref.size(); ++i) max_err = fmaxf(max_err, fabsf(__half2float(y_gpu[i]) - y_ref[i]));
    printf("%s: max abs err = %.4f  %s\n", name, max_err, max_err < 0.05f ? "PASS" : "FAIL");
    return max_err < 0.05f ? 0 : 1;
}

static void ref_q8_0(const uint8_t* blk, int b, float* row) {
    float d = __half2float(*(const __half*)blk);
    const int8_t* q = (const int8_t*)(blk + 2);
    for (int j = 0; j < 32; ++j) row[b * 32 + j] = d * (float)q[j];
}
static void ref_q4_K(const uint8_t* blk, int b, float* row) {
    float d = __half2float(*(const __half*)blk), dmin = __half2float(*(const __half*)(blk + 2));
    const uint8_t* scales = blk + 4; const uint8_t* qs = blk + 16;
    int is = 0, yi = 0; uint8_t sc, mn;
    for (int g = 0; g < 256; g += 64) {
        get_scale_min_k4(is + 0, scales, &sc, &mn); float d1 = d * sc, m1 = dmin * mn;
        get_scale_min_k4(is + 1, scales, &sc, &mn); float d2 = d * sc, m2 = dmin * mn;
        const uint8_t* q = qs + (g / 64) * 32;
        for (int l = 0; l < 32; ++l) row[b * 256 + yi++] = d1 * (float)(q[l] & 0xF) - m1;
        for (int l = 0; l < 32; ++l) row[b * 256 + yi++] = d2 * (float)(q[l] >> 4)  - m2;
        is += 2;
    }
}

int main() {
    srand(123);
    int rc = 0;
    rc |= run("fused_gemv Q8_0", 8, 34, 32, ref_q8_0);
    rc |= run("fused_gemv Q4_K", 12, 144, 256, ref_q4_K);
    return rc;
}

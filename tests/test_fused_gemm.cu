// Fused dequant-GEMV correctness: compare the fused kernel (reads native quantized
// weights) against a CPU dequant + GEMV reference, for Q8_0 and Q4_K.
#include "fused_gemm.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
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
        uint8_t* blk = W.data() + ((size_t)o * nb + b) * block_bytes;
        __half* d = (__half*)blk;
        float dv = 0.05f * (((o + b) % 7) + 1);
        if (type == 14) *(__half*)(blk + 208) = __float2half(dv * 0.05f);  // Q6_K d at offset 208 (int8 scales)
        else { d[0] = __float2half(dv);
               if (type == 12 || type == 13) d[1] = __float2half(0.03f * ((b % 5) + 1)); }  // Q4_K/Q5_K dmin
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

    // fp16 output has ~2^-11 relative precision, so compare RELATIVE error.
    float max_rel = 0;
    for (size_t i = 0; i < y_ref.size(); ++i) {
        float e = fabsf(__half2float(y_gpu[i]) - y_ref[i]) / (fabsf(y_ref[i]) + 1e-2f);
        max_rel = fmaxf(max_rel, e);
    }
    printf("%s: max rel err = %.4f  %s\n", name, max_rel, max_rel < 0.02f ? "PASS" : "FAIL");
    return max_rel < 0.02f ? 0 : 1;
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

static void ref_q5_K(const uint8_t* blk, int b, float* row) {
    float d = __half2float(*(const __half*)blk), dmin = __half2float(*(const __half*)(blk + 2));
    const uint8_t* scales = blk + 4; const uint8_t* qh = blk + 16; const uint8_t* ql = blk + 48;
    int is = 0, yi = 0; uint8_t sc, mn, u1 = 1, u2 = 2;
    for (int j = 0; j < 256; j += 64) {
        get_scale_min_k4(is + 0, scales, &sc, &mn); float d1 = d * sc, m1 = dmin * mn;
        get_scale_min_k4(is + 1, scales, &sc, &mn); float d2 = d * sc, m2 = dmin * mn;
        for (int l = 0; l < 32; ++l) row[b * 256 + yi++] = d1 * (float)((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
        for (int l = 0; l < 32; ++l) row[b * 256 + yi++] = d2 * (float)((ql[l] >> 4)  + ((qh[l] & u2) ? 16 : 0)) - m2;
        ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
    }
}
static void ref_q6_K(const uint8_t* blk, int b, float* row) {
    const uint8_t* ql = blk; const uint8_t* qh = blk + 128; const int8_t* sc = (const int8_t*)(blk + 192);
    float d = __half2float(*(const __half*)(blk + 208));
    for (int nn = 0; nn < 256; nn += 128) {
        const uint8_t* Ql = ql + (nn / 128) * 64; const uint8_t* Qh = qh + (nn / 128) * 32; const int8_t* Sc = sc + (nn / 128) * 8;
        for (int l = 0; l < 32; ++l) {
            int is = l / 16;
            int q1 = (int)((Ql[l + 0] & 0xF) | (((Qh[l] >> 0) & 3) << 4)) - 32;
            int q2 = (int)((Ql[l + 32] & 0xF) | (((Qh[l] >> 2) & 3) << 4)) - 32;
            int q3 = (int)((Ql[l + 0] >> 4)  | (((Qh[l] >> 4) & 3) << 4)) - 32;
            int q4 = (int)((Ql[l + 32] >> 4) | (((Qh[l] >> 6) & 3) << 4)) - 32;
            row[b * 256 + nn + l + 0]  = d * Sc[is + 0] * q1;
            row[b * 256 + nn + l + 32] = d * Sc[is + 2] * q2;
            row[b * 256 + nn + l + 64] = d * Sc[is + 4] * q3;
            row[b * 256 + nn + l + 96] = d * Sc[is + 6] * q4;
        }
    }
}

// Time a decode-shape GEMV (m=1) and report latency + achieved weight bandwidth. GEMV is
// bandwidth-bound, so GB/s near the device peak means the kernel is close to optimal.
static void bench(const char* name, uint32_t type, int block_bytes, int block_elems) {
    int n_in = 4096, n_out = 4096, m = 1, nb = n_in / block_elems;
    size_t wbytes = (size_t)n_out * nb * block_bytes;
    std::vector<uint8_t> W(wbytes); for (auto& b : W) b = rand() & 0xFF;
    std::vector<__half> x((size_t)m * n_in, __float2half(0.01f));
    uint8_t* dW; __half *dx, *dy;
    cudaMalloc(&dW, wbytes); cudaMalloc(&dx, x.size()*2); cudaMalloc(&dy, (size_t)m*n_out*2);
    cudaMemcpy(dW, W.data(), wbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x.data(), x.size()*2, cudaMemcpyHostToDevice);
    for (int i = 0; i < 10; ++i) fused_gemv(dW, dx, dy, m, n_out, n_in, type, 0);
    cudaDeviceSynchronize();
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    int iters = 200;
    cudaEventRecord(a);
    for (int i = 0; i < iters; ++i) fused_gemv(dW, dx, dy, m, n_out, n_in, type, 0);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms = 0; cudaEventElapsedTime(&ms, a, b);
    double us = ms * 1000.0 / iters;
    double gbps = wbytes / (us * 1e-6) / 1e9;
    printf("%s [%dx%d m=%d]: %.1f us/call, %.0f GB/s (weights)\n", name, n_out, n_in, m, us, gbps);
    cudaFree(dW); cudaFree(dx); cudaFree(dy);
}

int main(int argc, char** argv) {
    srand(123);
    if (argc > 1 && std::string(argv[1]) == "bench") {
        bench("Q8_0", 8, 34, 32);
        bench("Q4_K", 12, 144, 256);
        bench("Q5_K", 13, 176, 256);
        bench("Q6_K", 14, 210, 256);
        return 0;
    }
    int rc = 0;
    rc |= run("fused_gemv Q8_0", 8, 34, 32, ref_q8_0);
    rc |= run("fused_gemv Q4_K", 12, 144, 256, ref_q4_K);
    rc |= run("fused_gemv Q5_K", 13, 176, 256, ref_q5_K);
    rc |= run("fused_gemv Q6_K", 14, 210, 256, ref_q6_K);
    return rc;
}

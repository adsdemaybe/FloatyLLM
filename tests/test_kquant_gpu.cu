// GPU K-quant dequant (Q4_K/Q6_K) must match the host reference. RUN on a GPU.
#include "dequant.h"
#include "kquant.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

// Fill a K-quant block buffer: valid fp16 scales, pseudo-random quant bytes.
static void fill_bytes(std::vector<uint8_t>& b, int n_blocks, int block_bytes, int d_pos, int dmin_pos) {
    b.resize((size_t)n_blocks * block_bytes);
    for (size_t i = 0; i < b.size(); ++i) b[i] = (uint8_t)((i * 37 + 11) & 0xFF);
    uint16_t d = 0x2E66;      // ~0.1 in fp16
    uint16_t dm = 0x2000;     // ~0.03 in fp16
    for (int k = 0; k < n_blocks; ++k) {
        memcpy(&b[(size_t)k*block_bytes + d_pos], &d, 2);
        if (dmin_pos >= 0) memcpy(&b[(size_t)k*block_bytes + dmin_pos], &dm, 2);
    }
}

template <typename T>
static int check_gpu(const std::vector<uint8_t>& bytes, const std::vector<float>& href,
                     void (*gpu)(const T*, __half*, int, cudaStream_t), int n_blocks, const char* name) {
    T* d_b; __half* d_o;
    cudaMalloc(&d_b, bytes.size()); cudaMalloc(&d_o, (size_t)n_blocks*256*sizeof(__half));
    cudaMemcpy(d_b, bytes.data(), bytes.size(), cudaMemcpyHostToDevice);
    gpu(d_b, d_o, n_blocks, 0);
    cudaDeviceSynchronize();
    std::vector<__half> out((size_t)n_blocks*256);
    cudaMemcpy(out.data(), d_o, out.size()*sizeof(__half), cudaMemcpyDeviceToHost);
    cudaFree(d_b); cudaFree(d_o);
    int fails = 0; float maxd = 0;
    for (size_t i = 0; i < out.size(); ++i) {
        float g = __half2float(out[i]); maxd = fmaxf(maxd, fabsf(g - href[i]));
        if (fabsf(g - href[i]) > 5e-3f * (fabsf(href[i]) + 1.0f)) ++fails;
    }
    printf("%s: GPU vs host max_diff=%.5f %s\n", name, maxd, fails ? "FAIL" : "PASS");
    return fails;
}

int main() {
    const int n_blocks = 6, n = n_blocks * 256;
    int fails = 0;

    std::vector<uint8_t> q4(n); fill_bytes(q4, n_blocks, 144, 0, 2);   // Q4K: d@0, dmin@2
    std::vector<float> h4(n); dequant_q4_K_f32(q4.data(), n, h4.data());
    fails += check_gpu<BlockQ4K>(q4, h4, dequant_q4_K, n_blocks, "Q4_K");

    std::vector<uint8_t> q6(n); fill_bytes(q6, n_blocks, 210, 208, -1); // Q6K: d@208
    std::vector<float> h6(n); dequant_q6_K_f32(q6.data(), n, h6.data());
    fails += check_gpu<BlockQ6K>(q6, h6, dequant_q6_K, n_blocks, "Q6_K");

    if (fails) { printf("FAIL: %d mismatches\n", fails); return 1; }
    printf("PASS: GPU K-quant dequant matches host\n");
    return 0;
}

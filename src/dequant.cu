// Q8_0 dequant kernel. One CUDA block per Q8_0 block, one thread per value.
// x = d * q  (Q8_0 has no zero-point offset). See PLAN.md section 11.
#include "dequant.h"

__global__ void dequant_q8_0_kernel(const BlockQ80* blocks, __half* out, int n_blocks) {
    int b = blockIdx.x;
    if (b >= n_blocks) return;
    int j = threadIdx.x; // 0..QK8_0-1
    __half d = blocks[b].d;
    int8_t q = blocks[b].qs[j];
    out[b * QK8_0 + j] = __hmul(d, __int2half_rn((int)q));
}

void dequant_q8_0(const BlockQ80* d_blocks, __half* d_out, int n_blocks, cudaStream_t stream) {
    if (n_blocks <= 0) return;
    dequant_q8_0_kernel<<<n_blocks, QK8_0, 0, stream>>>(d_blocks, d_out, n_blocks);
}

// One CUDA block per Q4_0 block; thread j unpacks the nibble pair -> elements j, j+16.
__global__ void dequant_q4_0_kernel(const BlockQ40* blocks, __half* out, int n_blocks) {
    int b = blockIdx.x;
    if (b >= n_blocks) return;
    int j = threadIdx.x;  // 0..15
    float d = __half2float(blocks[b].d);
    uint8_t q = blocks[b].qs[j];
    out[b * 32 + j]      = __float2half(d * (float)((int)(q & 0x0F) - 8));
    out[b * 32 + j + 16] = __float2half(d * (float)((int)(q >> 4) - 8));
}

void dequant_q4_0(const BlockQ40* d_blocks, __half* d_out, int n_blocks, cudaStream_t stream) {
    if (n_blocks <= 0) return;
    dequant_q4_0_kernel<<<n_blocks, 16, 0, stream>>>(d_blocks, d_out, n_blocks);
}

// --- K-quants (one thread per 256-value super-block; blocks are independent) ---
__device__ __forceinline__ void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

__global__ void dequant_q4_K_kernel(const BlockQ4K* blocks, __half* y, int nb) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nb) return;
    const BlockQ4K& b = blocks[i];
    float d = __half2float(b.d), mn = __half2float(b.dmin);
    const uint8_t* q = b.qs;
    __half* yy = y + (size_t)i * 256;
    int is = 0, yi = 0; uint8_t sc, m;
    for (int j = 0; j < 256; j += 64) {
        get_scale_min_k4(is + 0, b.scales, &sc, &m); float d1 = d * sc, m1 = mn * m;
        get_scale_min_k4(is + 1, b.scales, &sc, &m); float d2 = d * sc, m2 = mn * m;
        for (int l = 0; l < 32; ++l) yy[yi++] = __float2half(d1 * (float)(q[l] & 0xF) - m1);
        for (int l = 0; l < 32; ++l) yy[yi++] = __float2half(d2 * (float)(q[l] >> 4) - m2);
        q += 32; is += 2;
    }
}

// Q5_K: like Q4_K but a 5th bit per weight from qh. Unsigned 5-bit (0..31),
// per-64 sub-block scale/min via get_scale_min_k4. See llama.cpp dequantize_row_q5_K.
__global__ void dequant_q5_K_kernel(const BlockQ5K* blocks, __half* y, int nb) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nb) return;
    const BlockQ5K& b = blocks[i];
    float d = __half2float(b.d), mn = __half2float(b.dmin);
    const uint8_t* ql = b.qs;
    const uint8_t* qh = b.qh;
    __half* yy = y + (size_t)i * 256;
    int is = 0, yi = 0; uint8_t sc, m; uint8_t u1 = 1, u2 = 2;
    for (int j = 0; j < 256; j += 64) {
        get_scale_min_k4(is + 0, b.scales, &sc, &m); float d1 = d * sc, m1 = mn * m;
        get_scale_min_k4(is + 1, b.scales, &sc, &m); float d2 = d * sc, m2 = mn * m;
        for (int l = 0; l < 32; ++l) yy[yi++] = __float2half(d1 * (float)((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1);
        for (int l = 0; l < 32; ++l) yy[yi++] = __float2half(d2 * (float)((ql[l] >> 4)  + ((qh[l] & u2) ? 16 : 0)) - m2);
        ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
    }
}

__global__ void dequant_q6_K_kernel(const BlockQ6K* blocks, __half* y, int nb) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nb) return;
    const BlockQ6K& b = blocks[i];
    float d = __half2float(b.d);
    __half* yy = y + (size_t)i * 256;
    for (int nn = 0; nn < 256; nn += 128) {
        const uint8_t* Ql = b.ql + (nn / 128) * 64;
        const uint8_t* Qh = b.qh + (nn / 128) * 32;
        const int8_t* Sc = b.scales + (nn / 128) * 8;
        for (int l = 0; l < 32; ++l) {
            int is = l / 16;
            int8_t q1 = (int8_t)((Ql[l + 0] & 0xF) | (((Qh[l] >> 0) & 3) << 4)) - 32;
            int8_t q2 = (int8_t)((Ql[l + 32] & 0xF) | (((Qh[l] >> 2) & 3) << 4)) - 32;
            int8_t q3 = (int8_t)((Ql[l + 0] >> 4) | (((Qh[l] >> 4) & 3) << 4)) - 32;
            int8_t q4 = (int8_t)((Ql[l + 32] >> 4) | (((Qh[l] >> 6) & 3) << 4)) - 32;
            yy[nn + l + 0]  = __float2half(d * Sc[is + 0] * q1);
            yy[nn + l + 32] = __float2half(d * Sc[is + 2] * q2);
            yy[nn + l + 64] = __float2half(d * Sc[is + 4] * q3);
            yy[nn + l + 96] = __float2half(d * Sc[is + 6] * q4);
        }
    }
}

void dequant_q4_K(const BlockQ4K* d_blocks, __half* d_out, int n_blocks, cudaStream_t stream) {
    if (n_blocks <= 0) return;
    int t = 128;
    dequant_q4_K_kernel<<<(n_blocks + t - 1) / t, t, 0, stream>>>(d_blocks, d_out, n_blocks);
}

void dequant_q5_K(const BlockQ5K* d_blocks, __half* d_out, int n_blocks, cudaStream_t stream) {
    if (n_blocks <= 0) return;
    int t = 128;
    dequant_q5_K_kernel<<<(n_blocks + t - 1) / t, t, 0, stream>>>(d_blocks, d_out, n_blocks);
}

void dequant_q6_K(const BlockQ6K* d_blocks, __half* d_out, int n_blocks, cudaStream_t stream) {
    if (n_blocks <= 0) return;
    int t = 128;
    dequant_q6_K_kernel<<<(n_blocks + t - 1) / t, t, 0, stream>>>(d_blocks, d_out, n_blocks);
}

// Q3_K super-block scale unpack: 12 packed bytes -> 16 6-bit scales in sc[0..15] (as int8,
// caller subtracts 32). Mirrors llama.cpp's aux shuffle. Shared by dequant + GEMV.
__device__ __forceinline__ void unpack_q3_scales(const uint8_t* s, int8_t* sc) {
    uint32_t a0 = s[0]|(s[1]<<8)|(s[2]<<16)|((uint32_t)s[3]<<24);
    uint32_t a1 = s[4]|(s[5]<<8)|(s[6]<<16)|((uint32_t)s[7]<<24);
    uint32_t a2 = s[8]|(s[9]<<8)|(s[10]<<16)|((uint32_t)s[11]<<24);
    const uint32_t km1 = 0x03030303u, km2 = 0x0f0f0f0fu;
    uint32_t aux[4];
    aux[2] = ((a0 >> 4) & km2) | (((a2 >> 4) & km1) << 4);
    aux[3] = ((a1 >> 4) & km2) | (((a2 >> 6) & km1) << 4);
    aux[0] = (a0 & km2) | (((a2 >> 0) & km1) << 4);
    aux[1] = (a1 & km2) | (((a2 >> 2) & km1) << 4);
    const int8_t* p = (const int8_t*)aux;
    #pragma unroll
    for (int i = 0; i < 16; ++i) sc[i] = p[i];
}

// Q3_K: 3-bit weights (2 bits in qs, 3rd bit in hmask), 16 6-bit sub-block scales.
// value(pos) = d * (scale-32) * ((qs bits) - (hmask bit ? 0 : 4)). See llama.cpp
// dequantize_row_q3_K. One thread per 256-value super-block.
__global__ void dequant_q3_K_kernel(const BlockQ3K* blocks, __half* y, int nb) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nb) return;
    const BlockQ3K& b = blocks[i];
    float d = __half2float(b.d);
    int8_t sc[16]; unpack_q3_scales(b.scales, sc);
    __half* yy = y + (size_t)i * 256;
    for (int h = 0; h < 2; ++h)
        for (int j = 0; j < 4; ++j)
            for (int r = 0; r < 32; ++r) {
                int qidx = h * 32 + r;
                int shift = 2 * j;
                uint8_t mbit = (uint8_t)(1u << (h * 4 + j));
                int sidx = h * 8 + j * 2 + (r >= 16 ? 1 : 0);
                int ql = (b.qs[qidx] >> shift) & 3;
                int hb = (b.hmask[r] & mbit) ? 0 : 4;
                yy[h * 128 + j * 32 + r] = __float2half(d * (float)(sc[sidx] - 32) * (float)(ql - hb));
            }
}

void dequant_q3_K(const BlockQ3K* d_blocks, __half* d_out, int n_blocks, cudaStream_t stream) {
    if (n_blocks <= 0) return;
    int t = 128;
    dequant_q3_K_kernel<<<(n_blocks + t - 1) / t, t, 0, stream>>>(d_blocks, d_out, n_blocks);
}

__global__ void transpose_kernel(const __half* src, __half* dst, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    int r = i / cols, c = i % cols;
    dst[(size_t)c * rows + r] = src[i];
}

void transpose_fp16(const __half* src, __half* dst, int rows, int cols, cudaStream_t stream) {
    if (rows <= 0 || cols <= 0) return;
    int n = rows * cols, t = 256;
    transpose_kernel<<<(n + t - 1) / t, t, 0, stream>>>(src, dst, rows, cols);
}

__global__ void f32_to_f16_kernel(const float* src, __half* dst, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

void dequant_to_fp16(const uint8_t* q, __half* out, uint32_t type, size_t n, cudaStream_t stream) {
    int t = 256;
    switch (type) {
        case 0:  // F32
            f32_to_f16_kernel<<<(n + t - 1) / t, t, 0, stream>>>((const float*)q, out, n); break;
        case 1:  // F16
            cudaMemcpyAsync(out, q, n * sizeof(__half), cudaMemcpyDeviceToDevice, stream); break;
        case 2:  dequant_q4_0((const BlockQ40*)q, out, (int)(n / 32), stream); break;   // Q4_0
        case 8:  dequant_q8_0((const BlockQ80*)q, out, (int)(n / 32), stream); break;   // Q8_0
        case 11: dequant_q3_K((const BlockQ3K*)q, out, (int)(n / 256), stream); break;  // Q3_K
        case 12: dequant_q4_K((const BlockQ4K*)q, out, (int)(n / 256), stream); break;  // Q4_K
        case 13: dequant_q5_K((const BlockQ5K*)q, out, (int)(n / 256), stream); break;  // Q5_K
        case 14: dequant_q6_K((const BlockQ6K*)q, out, (int)(n / 256), stream); break;  // Q6_K
        default: break;
    }
}

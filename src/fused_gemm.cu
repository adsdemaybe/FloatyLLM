#include "fused_gemm.h"

namespace {
constexpr int WARP = 32;
constexpr int WARPS = 8;                 // warps per block -> 256 threads, 8 rows/block
constexpr int THREADS = WARP * WARPS;

// Warp-level sum reduction (no shared memory, no __syncthreads).
__device__ __forceinline__ float warp_reduce(float v) {
    #pragma unroll
    for (int off = WARP / 2; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

// One WARP per (output row o, token tok): each lane strides over the row's quant blocks,
// accumulates a partial dot, then the warp reduces via __shfl. Activations loaded as
// half2. y[tok,o] = sum_i d_blk * q * x[tok,i].
__global__ void gemv_q8_0_kernel(const uint8_t* W, const __half* x, __half* y,
                                 int m, int n_out, int n_in) {
    int lane = threadIdx.x & (WARP - 1);
    int o = blockIdx.x * WARPS + (threadIdx.x >> 5);
    int tok = blockIdx.y;
    if (o >= n_out || tok >= m) return;
    int nb = n_in / 32;                                   // Q8_0 blocks per row
    const uint8_t* wrow = W + (size_t)o * nb * 34;        // 34 bytes/block
    const __half* xt = x + (size_t)tok * n_in;
    float acc = 0.0f;
    for (int b = lane; b < nb; b += WARP) {
        const uint8_t* blk = wrow + (size_t)b * 34;
        float d = __half2float(*(const __half*)blk);
        const int8_t* q = (const int8_t*)(blk + 2);
        const __half2* x2 = (const __half2*)(xt + b * 32);
        float s = 0.0f;
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            float2 xf = __half22float2(x2[j]);
            s += (float)q[2*j] * xf.x + (float)q[2*j+1] * xf.y;
        }
        acc += d * s;
    }
    acc = warp_reduce(acc);
    if (lane == 0) y[(size_t)tok * n_out + o] = __float2half(acc);
}

__device__ __forceinline__ void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

// Q4_K super-block: __half d, dmin; uint8_t scales[12]; uint8_t qs[128]. 144 B / 256 vals,
// laid out as 4 groups of 64 (low nibbles then high nibbles of a 32-byte slice).
// Warp per (row, token); each lane strides over 64-element GROUPS so all 32 lanes work
// (n_in/64 groups/row vs only n_in/256 super-blocks). x loaded as half2.
__global__ void gemv_q4_K_kernel(const uint8_t* W, const __half* x, __half* y,
                                 int m, int n_out, int n_in) {
    int lane = threadIdx.x & (WARP - 1);
    int o = blockIdx.x * WARPS + (threadIdx.x >> 5);
    int tok = blockIdx.y;
    if (o >= n_out || tok >= m) return;
    int nsb = n_in / 256, ng = n_in / 64;                 // super-blocks / 64-groups per row
    const uint8_t* wrow = W + (size_t)o * nsb * 144;
    const __half* xt = x + (size_t)tok * n_in;
    float acc = 0.0f;
    for (int g = lane; g < ng; g += WARP) {
        int sb = g >> 2, sub = g & 3;                     // super-block + which 64-group
        const uint8_t* blk = wrow + (size_t)sb * 144;
        float d = __half2float(*(const __half*)blk);
        float dmin = __half2float(*(const __half*)(blk + 2));
        const uint8_t* scales = blk + 4;
        const uint8_t* q = blk + 16 + sub * 32;
        uint8_t sc, mn;
        get_scale_min_k4(sub * 2 + 0, scales, &sc, &mn); float d1 = d * sc, m1 = dmin * mn;
        get_scale_min_k4(sub * 2 + 1, scales, &sc, &mn); float d2 = d * sc, m2 = dmin * mn;
        const __half2* xlo = (const __half2*)(xt + g * 64);
        const __half2* xhi = (const __half2*)(xt + g * 64 + 32);
        float s = 0.0f;
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            float2 a = __half22float2(xlo[j]), b = __half22float2(xhi[j]);
            s += (d1 * (float)(q[2*j]   & 0xF) - m1) * a.x + (d1 * (float)(q[2*j+1] & 0xF) - m1) * a.y;
            s += (d2 * (float)(q[2*j]   >> 4)  - m2) * b.x + (d2 * (float)(q[2*j+1] >> 4)  - m2) * b.y;
        }
        acc += s;
    }
    acc = warp_reduce(acc);
    if (lane == 0) y[(size_t)tok * n_out + o] = __float2half(acc);
}

// Q5_K super-block: __half d, dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128].
// 176 B / 256 vals. 5th bit per weight from qh; per-64 sub-block scale/min via
// get_scale_min_k4. Warp per (row, token); each lane strides over 256-val super-blocks
// and dots the whole block (mirrors dequant_q5_K, multiply-accumulate against x).
__global__ void gemv_q5_K_kernel(const uint8_t* W, const __half* x, __half* y,
                                 int m, int n_out, int n_in) {
    int lane = threadIdx.x & (WARP - 1);
    int o = blockIdx.x * WARPS + (threadIdx.x >> 5);
    int tok = blockIdx.y;
    if (o >= n_out || tok >= m) return;
    int nsb = n_in / 256;
    const uint8_t* wrow = W + (size_t)o * nsb * 176;
    const __half* xt = x + (size_t)tok * n_in;
    float acc = 0.0f;
    for (int sb = lane; sb < nsb; sb += WARP) {
        const uint8_t* blk = wrow + (size_t)sb * 176;
        float d = __half2float(*(const __half*)blk);
        float dmin = __half2float(*(const __half*)(blk + 2));
        const uint8_t* scales = blk + 4;
        const uint8_t* qh = blk + 16;
        const uint8_t* ql = blk + 48;
        const __half* xb = xt + (size_t)sb * 256;
        int is = 0; uint8_t u1 = 1, u2 = 2; int base = 0;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc, mn;
            get_scale_min_k4(is + 0, scales, &sc, &mn); float d1 = d * sc, m1 = dmin * mn;
            get_scale_min_k4(is + 1, scales, &sc, &mn); float d2 = d * sc, m2 = dmin * mn;
            for (int l = 0; l < 32; ++l) {
                float vlo = d1 * (float)((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
                float vhi = d2 * (float)((ql[l] >> 4)  + ((qh[l] & u2) ? 16 : 0)) - m2;
                acc += vlo * __half2float(xb[base + l]);
                acc += vhi * __half2float(xb[base + l + 32]);
            }
            ql += 32; is += 2; u1 <<= 2; u2 <<= 2; base += 64;
        }
    }
    acc = warp_reduce(acc);
    if (lane == 0) y[(size_t)tok * n_out + o] = __float2half(acc);
}

// Q6_K super-block: ql[128], qh[64], int8 scales[16], __half d. 210 B / 256 vals.
// Signed 6-bit q = (ql_nib | (qh_2b<<4)) - 32; x = d*scale[sub]*q. Warp per (row,
// token); lane strides super-blocks, dots the whole block (mirrors dequant_q6_K).
__global__ void gemv_q6_K_kernel(const uint8_t* W, const __half* x, __half* y,
                                 int m, int n_out, int n_in) {
    int lane = threadIdx.x & (WARP - 1);
    int o = blockIdx.x * WARPS + (threadIdx.x >> 5);
    int tok = blockIdx.y;
    if (o >= n_out || tok >= m) return;
    int nsb = n_in / 256;
    const uint8_t* wrow = W + (size_t)o * nsb * 210;
    const __half* xt = x + (size_t)tok * n_in;
    float acc = 0.0f;
    for (int sb = lane; sb < nsb; sb += WARP) {
        const uint8_t* blk = wrow + (size_t)sb * 210;
        const uint8_t* ql = blk;
        const uint8_t* qh = blk + 128;
        const int8_t* sc = (const int8_t*)(blk + 192);
        float d = __half2float(*(const __half*)(blk + 208));
        const __half* xb = xt + (size_t)sb * 256;
        for (int nn = 0; nn < 256; nn += 128) {
            const uint8_t* Ql = ql + (nn / 128) * 64;
            const uint8_t* Qh = qh + (nn / 128) * 32;
            const int8_t* Sc = sc + (nn / 128) * 8;
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                int q1 = (int)((Ql[l + 0] & 0xF) | (((Qh[l] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((Ql[l + 32] & 0xF) | (((Qh[l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((Ql[l + 0] >> 4)  | (((Qh[l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((Ql[l + 32] >> 4) | (((Qh[l] >> 6) & 3) << 4)) - 32;
                acc += d * Sc[is + 0] * q1 * __half2float(xb[nn + l + 0]);
                acc += d * Sc[is + 2] * q2 * __half2float(xb[nn + l + 32]);
                acc += d * Sc[is + 4] * q3 * __half2float(xb[nn + l + 64]);
                acc += d * Sc[is + 6] * q4 * __half2float(xb[nn + l + 96]);
            }
        }
    }
    acc = warp_reduce(acc);
    if (lane == 0) y[(size_t)tok * n_out + o] = __float2half(acc);
}
}  // namespace

void fused_gemv_q8_0(const uint8_t* W, const __half* x, __half* y, int m, int n_out, int n_in, cudaStream_t stream) {
    if (m <= 0 || n_out <= 0 || n_in <= 0) return;
    dim3 grid((n_out + WARPS - 1) / WARPS, m);
    gemv_q8_0_kernel<<<grid, THREADS, 0, stream>>>(W, x, y, m, n_out, n_in);
}

void fused_gemv_q4_K(const uint8_t* W, const __half* x, __half* y, int m, int n_out, int n_in, cudaStream_t stream) {
    if (m <= 0 || n_out <= 0 || n_in <= 0) return;
    dim3 grid((n_out + WARPS - 1) / WARPS, m);
    gemv_q4_K_kernel<<<grid, THREADS, 0, stream>>>(W, x, y, m, n_out, n_in);
}

void fused_gemv_q5_K(const uint8_t* W, const __half* x, __half* y, int m, int n_out, int n_in, cudaStream_t stream) {
    if (m <= 0 || n_out <= 0 || n_in <= 0) return;
    dim3 grid((n_out + WARPS - 1) / WARPS, m);
    gemv_q5_K_kernel<<<grid, THREADS, 0, stream>>>(W, x, y, m, n_out, n_in);
}

void fused_gemv_q6_K(const uint8_t* W, const __half* x, __half* y, int m, int n_out, int n_in, cudaStream_t stream) {
    if (m <= 0 || n_out <= 0 || n_in <= 0) return;
    dim3 grid((n_out + WARPS - 1) / WARPS, m);
    gemv_q6_K_kernel<<<grid, THREADS, 0, stream>>>(W, x, y, m, n_out, n_in);
}

bool fused_gemv(const uint8_t* W, const __half* x, __half* y, int m, int n_out, int n_in,
                uint32_t ggml_type, cudaStream_t stream) {
    switch (ggml_type) {
        case 8:  fused_gemv_q8_0(W, x, y, m, n_out, n_in, stream); return true;   // Q8_0
        case 12: fused_gemv_q4_K(W, x, y, m, n_out, n_in, stream); return true;   // Q4_K
        case 13: fused_gemv_q5_K(W, x, y, m, n_out, n_in, stream); return true;   // Q5_K
        case 14: fused_gemv_q6_K(W, x, y, m, n_out, n_in, stream); return true;   // Q6_K
        default: return false;
    }
}

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

// Q4_K super-block: __half d, dmin; uint8_t scales[12]; uint8_t qs[128]. 144 B / 256 vals.
// Warp per (row, token); each lane strides over super-blocks.
__global__ void gemv_q4_K_kernel(const uint8_t* W, const __half* x, __half* y,
                                 int m, int n_out, int n_in) {
    int lane = threadIdx.x & (WARP - 1);
    int o = blockIdx.x * WARPS + (threadIdx.x >> 5);
    int tok = blockIdx.y;
    if (o >= n_out || tok >= m) return;
    int nsb = n_in / 256;                                 // super-blocks per row
    const uint8_t* wrow = W + (size_t)o * nsb * 144;
    const __half* xt = x + (size_t)tok * n_in;
    float acc = 0.0f;
    for (int sb = lane; sb < nsb; sb += WARP) {
        const uint8_t* blk = wrow + (size_t)sb * 144;
        float d = __half2float(*(const __half*)blk);
        float dmin = __half2float(*(const __half*)(blk + 2));
        const uint8_t* scales = blk + 4;
        const uint8_t* qs = blk + 16;
        const __half* xsb = xt + sb * 256;
        int is = 0, yi = 0; uint8_t sc, mn;
        float s = 0.0f;
        for (int gg = 0; gg < 256; gg += 64) {
            get_scale_min_k4(is + 0, scales, &sc, &mn); float d1 = d * sc, m1 = dmin * mn;
            get_scale_min_k4(is + 1, scales, &sc, &mn); float d2 = d * sc, m2 = dmin * mn;
            const uint8_t* q = qs + (gg / 64) * 32;
            #pragma unroll
            for (int l = 0; l < 32; ++l) s += (d1 * (float)(q[l] & 0xF) - m1) * __half2float(xsb[yi++]);
            #pragma unroll
            for (int l = 0; l < 32; ++l) s += (d2 * (float)(q[l] >> 4)  - m2) * __half2float(xsb[yi++]);
            is += 2;
        }
        acc += s;
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

bool fused_gemv(const uint8_t* W, const __half* x, __half* y, int m, int n_out, int n_in,
                uint32_t ggml_type, cudaStream_t stream) {
    switch (ggml_type) {
        case 8:  fused_gemv_q8_0(W, x, y, m, n_out, n_in, stream); return true;   // Q8_0
        case 12: fused_gemv_q4_K(W, x, y, m, n_out, n_in, stream); return true;   // Q4_K
        default: return false;
    }
}

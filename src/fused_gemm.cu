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

// Q3_K super-block scale unpack: 12 packed bytes -> 16 6-bit scales (as int8; caller -32).
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

// Q2_K: scales[16], qs[64], __half d, dmin. 84 B / 256. Warp per (row, token); lane l
// handles element l of each 32-slice (h in {0,1}, j in 0..3 -> 8 values/lane), coalesced.
// value = d*(scale nibble)*q2 - dmin*(min nibble).
__global__ void gemv_q2_K_kernel(const uint8_t* W, const __half* x, __half* y,
                                 int m, int n_out, int n_in) {
    int lane = threadIdx.x & (WARP - 1);
    int o = blockIdx.x * WARPS + (threadIdx.x >> 5);
    int tok = blockIdx.y;
    if (o >= n_out || tok >= m) return;
    int nsb = n_in / 256;
    const uint8_t* wrow = W + (size_t)o * nsb * 84;
    const __half* xt = x + (size_t)tok * n_in;
    float acc = 0.0f;
    for (int sb = 0; sb < nsb; ++sb) {
        const uint8_t* blk = wrow + (size_t)sb * 84;
        const uint8_t* scales = blk;
        const uint8_t* qs = blk + 16;
        float d = __half2float(*(const __half*)(blk + 80));
        float dmin = __half2float(*(const __half*)(blk + 82));
        const __half* xb = xt + (size_t)sb * 256;
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            uint8_t qb = qs[h * 32 + lane];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint8_t sc = scales[h * 8 + j * 2 + (lane >= 16 ? 1 : 0)];
                float dl = d * (float)(sc & 0xF), ml = dmin * (float)(sc >> 4);
                int q2 = (qb >> (2 * j)) & 3;
                acc += (dl * (float)q2 - ml) * __half2float(xb[h * 128 + j * 32 + lane]);
            }
        }
    }
    acc = warp_reduce(acc);
    if (lane == 0) y[(size_t)tok * n_out + o] = __float2half(acc);
}

// Q3_K: hmask[32], qs[64], scales[12], __half d. 110 B / 256. Warp per (row, token);
// lane l handles element l of each 32-slice (h in {0,1}, j in 0..3 -> 8 values/lane),
// coalesced. value = d*(scale-32)*((qs 2 bits) - (hmask bit ? 0 : 4)).
__global__ void gemv_q3_K_kernel(const uint8_t* W, const __half* x, __half* y,
                                 int m, int n_out, int n_in) {
    int lane = threadIdx.x & (WARP - 1);
    int o = blockIdx.x * WARPS + (threadIdx.x >> 5);
    int tok = blockIdx.y;
    if (o >= n_out || tok >= m) return;
    int nsb = n_in / 256;
    const uint8_t* wrow = W + (size_t)o * nsb * 110;
    const __half* xt = x + (size_t)tok * n_in;
    float acc = 0.0f;
    for (int sb = 0; sb < nsb; ++sb) {
        const uint8_t* blk = wrow + (size_t)sb * 110;
        const uint8_t* hmask = blk;
        const uint8_t* qs = blk + 32;
        float d = __half2float(*(const __half*)(blk + 108));
        int8_t sc[16]; unpack_q3_scales(blk + 96, sc);
        const __half* xb = xt + (size_t)sb * 256;
        uint8_t hml = hmask[lane];
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            uint8_t qb = qs[h * 32 + lane];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint8_t mbit = (uint8_t)(1u << (h * 4 + j));
                int sidx = h * 8 + j * 2 + (lane >= 16 ? 1 : 0);
                int ql = (qb >> (2 * j)) & 3;
                int hb = (hml & mbit) ? 0 : 4;
                acc += d * (float)(sc[sidx] - 32) * (float)(ql - hb) * __half2float(xb[h * 128 + j * 32 + lane]);
            }
        }
    }
    acc = warp_reduce(acc);
    if (lane == 0) y[(size_t)tok * n_out + o] = __float2half(acc);
}

// Q4_K super-block: __half d, dmin; uint8_t scales[12]; uint8_t qs[128]. 144 B / 256 vals.
// Warp per (row, token). The warp splits into 4 lane-groups of 8 (g = lane/8 = which
// 64-value sub-block); within a group each lane reads a uint32 (4 qs bytes) so the whole
// warp reads the 128-byte qs region in ONE coalesced transaction (vs 4 byte-reads).
// Each lane then does 8 products (4 low-nibble + 4 high-nibble elements). Lifts K-quant
// GEMV toward the Q8_0 bandwidth (byte-at-a-time unpack was transaction-bound < DRAM peak).
__global__ void gemv_q4_K_kernel(const uint8_t* W, const __half* x, __half* y,
                                 int m, int n_out, int n_in) {
    int lane = threadIdx.x & (WARP - 1);
    int o = blockIdx.x * WARPS + (threadIdx.x >> 5);
    int tok = blockIdx.y;
    if (o >= n_out || tok >= m) return;
    int nsb = n_in / 256;
    const uint8_t* wrow = W + (size_t)o * nsb * 144;
    const __half* xt = x + (size_t)tok * n_in;
    int g = lane >> 3;          // 0..3 : which 64-value sub-block
    int bl = (lane & 7) * 4;    // 0,4,..28 : byte offset within the sub-block's 32 qs bytes
    float acc = 0.0f;
    for (int sb = 0; sb < nsb; ++sb) {
        const uint8_t* blk = wrow + (size_t)sb * 144;
        float d = __half2float(*(const __half*)blk);
        float dmin = __half2float(*(const __half*)(blk + 2));
        const uint8_t* scales = blk + 4;
        const __half* xb = xt + (size_t)sb * 256 + g * 64;   // this sub-block's 64 activations
        uint8_t sc, mn;
        get_scale_min_k4(g * 2 + 0, scales, &sc, &mn); float d1 = d * sc, m1 = dmin * mn;
        get_scale_min_k4(g * 2 + 1, scales, &sc, &mn); float d2 = d * sc, m2 = dmin * mn;
        uint32_t u = *(const uint32_t*)(blk + 16 + g * 32 + bl);
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            uint8_t q = (u >> (8 * k)) & 0xFF;
            acc += (d1 * (float)(q & 0xF) - m1) * __half2float(xb[bl + k]);
            acc += (d2 * (float)(q >> 4)  - m2) * __half2float(xb[bl + 32 + k]);
        }
    }
    acc = warp_reduce(acc);
    if (lane == 0) y[(size_t)tok * n_out + o] = __float2half(acc);
}

// Q5_K super-block: __half d, dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128].
// 176 B / 256 vals. Warp per (row, token); the 32 lanes cooperate on ONE super-block at
// a time (lane l handles element l of each 32-slice) so weight/activation reads coalesce.
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
    for (int sb = 0; sb < nsb; ++sb) {
        const uint8_t* blk = wrow + (size_t)sb * 176;
        float d = __half2float(*(const __half*)blk);
        float dmin = __half2float(*(const __half*)(blk + 2));
        const uint8_t* scales = blk + 4;
        const uint8_t* qh = blk + 16;      // 32 bytes: bit (2*jj)=lo 5th bit, (2*jj+1)=hi
        const uint8_t* ql = blk + 48;      // 128 bytes, 32 per 64-sub-block
        const __half* xb = xt + (size_t)sb * 256;
        uint8_t qhl = qh[lane];
        #pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
            uint8_t sc, mn;
            get_scale_min_k4(jj * 2 + 0, scales, &sc, &mn); float d1 = d * sc, m1 = dmin * mn;
            get_scale_min_k4(jj * 2 + 1, scales, &sc, &mn); float d2 = d * sc, m2 = dmin * mn;
            uint8_t q = ql[jj * 32 + lane];
            float vlo = d1 * (float)((q & 0xF) + ((qhl & (1 << (2*jj)))     ? 16 : 0)) - m1;
            float vhi = d2 * (float)((q >> 4)  + ((qhl & (2 << (2*jj)))     ? 16 : 0)) - m2;
            acc += vlo * __half2float(xb[jj * 64 + lane]);
            acc += vhi * __half2float(xb[jj * 64 + 32 + lane]);
        }
    }
    acc = warp_reduce(acc);
    if (lane == 0) y[(size_t)tok * n_out + o] = __float2half(acc);
}

// Q6_K super-block: ql[128], qh[64], int8 scales[16], __half d. 210 B / 256 vals.
// Signed 6-bit q = (ql_nib | (qh_2b<<4)) - 32; x = d*scale[sub]*q. Warp per (row, token);
// the 32 lanes cooperate on ONE super-block at a time (lane l handles element l of each
// 32-slice) so weight/activation reads coalesce (vs the 210 B/lane strided version).
__global__ void gemv_q6_K_kernel(const uint8_t* W, const __half* x, __half* y,
                                 int m, int n_out, int n_in) {
    int lane = threadIdx.x & (WARP - 1);
    int o = blockIdx.x * WARPS + (threadIdx.x >> 5);
    int tok = blockIdx.y;
    if (o >= n_out || tok >= m) return;
    int nsb = n_in / 256;
    const uint8_t* wrow = W + (size_t)o * nsb * 210;
    const __half* xt = x + (size_t)tok * n_in;
    int is = lane / 16;   // scale sub-index within a 128-half
    float acc = 0.0f;
    for (int sb = 0; sb < nsb; ++sb) {
        const uint8_t* blk = wrow + (size_t)sb * 210;
        const uint8_t* ql = blk;
        const uint8_t* qh = blk + 128;
        const int8_t* sc = (const int8_t*)(blk + 192);
        float d = __half2float(*(const __half*)(blk + 208));
        const __half* xb = xt + (size_t)sb * 256;
        #pragma unroll
        for (int nn = 0; nn < 256; nn += 128) {
            const uint8_t* Ql = ql + (nn / 128) * 64;
            const uint8_t* Qh = qh + (nn / 128) * 32;
            const int8_t* Sc = sc + (nn / 128) * 8;
            uint8_t ql0 = Ql[lane], ql1 = Ql[lane + 32], qhb = Qh[lane];
            int q1 = (int)((ql0 & 0xF) | (((qhb >> 0) & 3) << 4)) - 32;
            int q2 = (int)((ql1 & 0xF) | (((qhb >> 2) & 3) << 4)) - 32;
            int q3 = (int)((ql0 >> 4)  | (((qhb >> 4) & 3) << 4)) - 32;
            int q4 = (int)((ql1 >> 4)  | (((qhb >> 6) & 3) << 4)) - 32;
            acc += d * Sc[is + 0] * q1 * __half2float(xb[nn + lane + 0]);
            acc += d * Sc[is + 2] * q2 * __half2float(xb[nn + lane + 32]);
            acc += d * Sc[is + 4] * q3 * __half2float(xb[nn + lane + 64]);
            acc += d * Sc[is + 6] * q4 * __half2float(xb[nn + lane + 96]);
        }
    }
    acc = warp_reduce(acc);
    if (lane == 0) y[(size_t)tok * n_out + o] = __float2half(acc);
}
// ===================== Fused dequant-GEMM (prefill: read W once, all m tokens) =====================
// Decode uses fused GEMV (one token) which re-reads W per token -> for prefill (m tokens) that is
// m full weight reads. These GEMM kernels read each weight ONCE and accumulate into TM token
// accumulators, so a prefill reads W ceil(m/TM) times instead of m. Same per-type unpack as the
// GEMV kernels above (verified against CPU ref); only the inner x-multiply is batched over tokens.
constexpr int TILE_M = 8;

// acc[t] += w * x[token (t0+t), pos] for the TM tokens in this tile (masked to m).
template<int TM>
__device__ __forceinline__ void mm_accum(float* acc, const __half* x, int n_in, int t0, int m,
                                         int pos, float w) {
    #pragma unroll
    for (int t = 0; t < TM; ++t) { int tk = t0 + t; if (tk < m) acc[t] += w * __half2float(x[(size_t)tk * n_in + pos]); }
}
template<int TM>
__device__ __forceinline__ void mm_store(float* acc, __half* y, int n_out, int t0, int m, int o, int lane) {
    #pragma unroll
    for (int t = 0; t < TM; ++t) { float a = warp_reduce(acc[t]); if (lane == 0) { int tk = t0 + t; if (tk < m) y[(size_t)tk * n_out + o] = __float2half(a); } }
}

template<int TM>
__global__ void gemm_q4_K_kernel(const uint8_t* W, const __half* x, __half* y, int m, int n_out, int n_in) {
    int lane = threadIdx.x & (WARP - 1); int o = blockIdx.x * WARPS + (threadIdx.x >> 5); int t0 = blockIdx.y * TM;
    if (o >= n_out) return;
    int nsb = n_in / 256; const uint8_t* wrow = W + (size_t)o * nsb * 144;
    int g = lane >> 3, bl = (lane & 7) * 4;
    float acc[TM]; for (int t = 0; t < TM; ++t) acc[t] = 0.f;
    for (int sb = 0; sb < nsb; ++sb) {
        const uint8_t* blk = wrow + (size_t)sb * 144;
        float d = __half2float(*(const __half*)blk), dmin = __half2float(*(const __half*)(blk + 2));
        const uint8_t* scales = blk + 4; uint8_t sc, mn;
        get_scale_min_k4(g * 2 + 0, scales, &sc, &mn); float d1 = d * sc, m1 = dmin * mn;
        get_scale_min_k4(g * 2 + 1, scales, &sc, &mn); float d2 = d * sc, m2 = dmin * mn;
        uint32_t u = *(const uint32_t*)(blk + 16 + g * 32 + bl);
        int base = sb * 256 + g * 64;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            uint8_t q = (u >> (8 * k)) & 0xFF;
            mm_accum<TM>(acc, x, n_in, t0, m, base + bl + k,      d1 * (float)(q & 0xF) - m1);
            mm_accum<TM>(acc, x, n_in, t0, m, base + bl + 32 + k, d2 * (float)(q >> 4)  - m2);
        }
    }
    mm_store<TM>(acc, y, n_out, t0, m, o, lane);
}

template<int TM>
__global__ void gemm_q6_K_kernel(const uint8_t* W, const __half* x, __half* y, int m, int n_out, int n_in) {
    int lane = threadIdx.x & (WARP - 1); int o = blockIdx.x * WARPS + (threadIdx.x >> 5); int t0 = blockIdx.y * TM;
    if (o >= n_out) return;
    int nsb = n_in / 256; const uint8_t* wrow = W + (size_t)o * nsb * 210; int is = lane / 16;
    float acc[TM]; for (int t = 0; t < TM; ++t) acc[t] = 0.f;
    for (int sb = 0; sb < nsb; ++sb) {
        const uint8_t* blk = wrow + (size_t)sb * 210;
        const uint8_t* ql = blk; const uint8_t* qh = blk + 128; const int8_t* sc = (const int8_t*)(blk + 192);
        float d = __half2float(*(const __half*)(blk + 208)); int sbase = sb * 256;
        #pragma unroll
        for (int nn = 0; nn < 256; nn += 128) {
            const uint8_t* Ql = ql + (nn / 128) * 64; const uint8_t* Qh = qh + (nn / 128) * 32; const int8_t* Sc = sc + (nn / 128) * 8;
            uint8_t ql0 = Ql[lane], ql1 = Ql[lane + 32], qhb = Qh[lane];
            int q1 = (int)((ql0 & 0xF) | (((qhb >> 0) & 3) << 4)) - 32;
            int q2 = (int)((ql1 & 0xF) | (((qhb >> 2) & 3) << 4)) - 32;
            int q3 = (int)((ql0 >> 4)  | (((qhb >> 4) & 3) << 4)) - 32;
            int q4 = (int)((ql1 >> 4)  | (((qhb >> 6) & 3) << 4)) - 32;
            mm_accum<TM>(acc, x, n_in, t0, m, sbase + nn + lane + 0,  d * Sc[is + 0] * q1);
            mm_accum<TM>(acc, x, n_in, t0, m, sbase + nn + lane + 32, d * Sc[is + 2] * q2);
            mm_accum<TM>(acc, x, n_in, t0, m, sbase + nn + lane + 64, d * Sc[is + 4] * q3);
            mm_accum<TM>(acc, x, n_in, t0, m, sbase + nn + lane + 96, d * Sc[is + 6] * q4);
        }
    }
    mm_store<TM>(acc, y, n_out, t0, m, o, lane);
}

template<int TM>
__global__ void gemm_q5_K_kernel(const uint8_t* W, const __half* x, __half* y, int m, int n_out, int n_in) {
    int lane = threadIdx.x & (WARP - 1); int o = blockIdx.x * WARPS + (threadIdx.x >> 5); int t0 = blockIdx.y * TM;
    if (o >= n_out) return;
    int nsb = n_in / 256; const uint8_t* wrow = W + (size_t)o * nsb * 176;
    float acc[TM]; for (int t = 0; t < TM; ++t) acc[t] = 0.f;
    for (int sb = 0; sb < nsb; ++sb) {
        const uint8_t* blk = wrow + (size_t)sb * 176;
        float d = __half2float(*(const __half*)blk), dmin = __half2float(*(const __half*)(blk + 2));
        const uint8_t* scales = blk + 4; const uint8_t* qh = blk + 16; const uint8_t* ql = blk + 48;
        uint8_t qhl = qh[lane]; int sbase = sb * 256;
        #pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
            uint8_t sc, mn;
            get_scale_min_k4(jj * 2 + 0, scales, &sc, &mn); float d1 = d * sc, m1 = dmin * mn;
            get_scale_min_k4(jj * 2 + 1, scales, &sc, &mn); float d2 = d * sc, m2 = dmin * mn;
            uint8_t q = ql[jj * 32 + lane];
            mm_accum<TM>(acc, x, n_in, t0, m, sbase + jj * 64 + lane,      d1 * (float)((q & 0xF) + ((qhl & (1 << (2*jj))) ? 16 : 0)) - m1);
            mm_accum<TM>(acc, x, n_in, t0, m, sbase + jj * 64 + 32 + lane, d2 * (float)((q >> 4)  + ((qhl & (2 << (2*jj))) ? 16 : 0)) - m2);
        }
    }
    mm_store<TM>(acc, y, n_out, t0, m, o, lane);
}

template<int TM>
__global__ void gemm_q3_K_kernel(const uint8_t* W, const __half* x, __half* y, int m, int n_out, int n_in) {
    int lane = threadIdx.x & (WARP - 1); int o = blockIdx.x * WARPS + (threadIdx.x >> 5); int t0 = blockIdx.y * TM;
    if (o >= n_out) return;
    int nsb = n_in / 256; const uint8_t* wrow = W + (size_t)o * nsb * 110;
    float acc[TM]; for (int t = 0; t < TM; ++t) acc[t] = 0.f;
    for (int sb = 0; sb < nsb; ++sb) {
        const uint8_t* blk = wrow + (size_t)sb * 110; const uint8_t* hmask = blk; const uint8_t* qs = blk + 32;
        float d = __half2float(*(const __half*)(blk + 108)); int8_t scl[16]; unpack_q3_scales(blk + 96, scl);
        uint8_t hml = hmask[lane]; int sbase = sb * 256;
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            uint8_t qb = qs[h * 32 + lane];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint8_t mbit = (uint8_t)(1u << (h * 4 + j));
                int sidx = h * 8 + j * 2 + (lane >= 16 ? 1 : 0);
                int ql = (qb >> (2 * j)) & 3; int hb = (hml & mbit) ? 0 : 4;
                mm_accum<TM>(acc, x, n_in, t0, m, sbase + h * 128 + j * 32 + lane, d * (float)(scl[sidx] - 32) * (float)(ql - hb));
            }
        }
    }
    mm_store<TM>(acc, y, n_out, t0, m, o, lane);
}

template<int TM>
__global__ void gemm_q2_K_kernel(const uint8_t* W, const __half* x, __half* y, int m, int n_out, int n_in) {
    int lane = threadIdx.x & (WARP - 1); int o = blockIdx.x * WARPS + (threadIdx.x >> 5); int t0 = blockIdx.y * TM;
    if (o >= n_out) return;
    int nsb = n_in / 256; const uint8_t* wrow = W + (size_t)o * nsb * 84;
    float acc[TM]; for (int t = 0; t < TM; ++t) acc[t] = 0.f;
    for (int sb = 0; sb < nsb; ++sb) {
        const uint8_t* blk = wrow + (size_t)sb * 84; const uint8_t* scales = blk; const uint8_t* qs = blk + 16;
        float d = __half2float(*(const __half*)(blk + 80)), dmin = __half2float(*(const __half*)(blk + 82)); int sbase = sb * 256;
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            uint8_t qb = qs[h * 32 + lane];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint8_t sc = scales[h * 8 + j * 2 + (lane >= 16 ? 1 : 0)];
                float dl = d * (float)(sc & 0xF), ml = dmin * (float)(sc >> 4);
                mm_accum<TM>(acc, x, n_in, t0, m, sbase + h * 128 + j * 32 + lane, dl * (float)((qb >> (2 * j)) & 3) - ml);
            }
        }
    }
    mm_store<TM>(acc, y, n_out, t0, m, o, lane);
}

template<int TM>
__global__ void gemm_q8_0_kernel(const uint8_t* W, const __half* x, __half* y, int m, int n_out, int n_in) {
    int lane = threadIdx.x & (WARP - 1); int o = blockIdx.x * WARPS + (threadIdx.x >> 5); int t0 = blockIdx.y * TM;
    if (o >= n_out) return;
    int nb = n_in / 32; const uint8_t* wrow = W + (size_t)o * nb * 34;
    float acc[TM]; for (int t = 0; t < TM; ++t) acc[t] = 0.f;
    for (int b = lane; b < nb; b += WARP) {
        const uint8_t* blk = wrow + (size_t)b * 34; float d = __half2float(*(const __half*)blk);
        const int8_t* q = (const int8_t*)(blk + 2); int base = b * 32;
        #pragma unroll
        for (int j = 0; j < 32; ++j) mm_accum<TM>(acc, x, n_in, t0, m, base + j, d * (float)q[j]);
    }
    mm_store<TM>(acc, y, n_out, t0, m, o, lane);
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

void fused_gemv_q2_K(const uint8_t* W, const __half* x, __half* y, int m, int n_out, int n_in, cudaStream_t stream) {
    if (m <= 0 || n_out <= 0 || n_in <= 0) return;
    dim3 grid((n_out + WARPS - 1) / WARPS, m);
    gemv_q2_K_kernel<<<grid, THREADS, 0, stream>>>(W, x, y, m, n_out, n_in);
}

void fused_gemv_q3_K(const uint8_t* W, const __half* x, __half* y, int m, int n_out, int n_in, cudaStream_t stream) {
    if (m <= 0 || n_out <= 0 || n_in <= 0) return;
    dim3 grid((n_out + WARPS - 1) / WARPS, m);
    gemv_q3_K_kernel<<<grid, THREADS, 0, stream>>>(W, x, y, m, n_out, n_in);
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
        case 10: fused_gemv_q2_K(W, x, y, m, n_out, n_in, stream); return true;   // Q2_K
        case 11: fused_gemv_q3_K(W, x, y, m, n_out, n_in, stream); return true;   // Q3_K
        case 12: fused_gemv_q4_K(W, x, y, m, n_out, n_in, stream); return true;   // Q4_K
        case 13: fused_gemv_q5_K(W, x, y, m, n_out, n_in, stream); return true;   // Q5_K
        case 14: fused_gemv_q6_K(W, x, y, m, n_out, n_in, stream); return true;   // Q6_K
        default: return false;
    }
}

// Fused dequant-GEMM: y[m,n_out] = x[m,n_in] * dequant(W[n_out,n_in]), reading each weight
// ONCE and accumulating over TILE_M token columns. For prefill (m>1) this reads W ceil(m/TILE_M)
// times vs m times for repeated GEMV. Returns false if the type has no fused GEMM kernel.
bool fused_gemm(const uint8_t* W, const __half* x, __half* y, int m, int n_out, int n_in,
                uint32_t ggml_type, cudaStream_t stream) {
    if (m <= 0 || n_out <= 0 || n_in <= 0) return true;
    dim3 grid((n_out + WARPS - 1) / WARPS, (m + TILE_M - 1) / TILE_M);
    switch (ggml_type) {
        case 8:  gemm_q8_0_kernel<TILE_M><<<grid, THREADS, 0, stream>>>(W, x, y, m, n_out, n_in); return true;
        case 10: gemm_q2_K_kernel<TILE_M><<<grid, THREADS, 0, stream>>>(W, x, y, m, n_out, n_in); return true;
        case 11: gemm_q3_K_kernel<TILE_M><<<grid, THREADS, 0, stream>>>(W, x, y, m, n_out, n_in); return true;
        case 12: gemm_q4_K_kernel<TILE_M><<<grid, THREADS, 0, stream>>>(W, x, y, m, n_out, n_in); return true;
        case 13: gemm_q5_K_kernel<TILE_M><<<grid, THREADS, 0, stream>>>(W, x, y, m, n_out, n_in); return true;
        case 14: gemm_q6_K_kernel<TILE_M><<<grid, THREADS, 0, stream>>>(W, x, y, m, n_out, n_in); return true;
        default: return false;
    }
}

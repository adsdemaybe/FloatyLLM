// FloatyLLM Metal kernels: MSL ports of the CUDA compute kernels, for Apple Silicon
// (M-series) unified memory -- the same "read quantized weights straight from unified RAM"
// streaming regime as GB10. Fused K-quant GEMV (decode), plus rmsnorm / rope / softmax /
// elementwise / embed / argmax. One SIMD-group (32 lanes on Apple GPUs) per output row,
// reduced with simd_sum -- the Metal analogue of the CUDA warp + __shfl reduction.
#include <metal_stdlib>
using namespace metal;

constant int WARP = 32;

// --- K-quant scale unpack helpers (identical math to the CUDA kernels) ---
inline void get_scale_min_k4(int j, device const uchar* q, thread uchar& d, thread uchar& m) {
    if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
    else { d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}
inline void unpack_q3_scales(device const uchar* s, thread char* sc) {
    uint a0 = s[0]|(uint(s[1])<<8)|(uint(s[2])<<16)|(uint(s[3])<<24);
    uint a1 = s[4]|(uint(s[5])<<8)|(uint(s[6])<<16)|(uint(s[7])<<24);
    uint a2 = s[8]|(uint(s[9])<<8)|(uint(s[10])<<16)|(uint(s[11])<<24);
    const uint km1 = 0x03030303u, km2 = 0x0f0f0f0fu;
    uint aux[4];
    aux[2] = ((a0 >> 4) & km2) | (((a2 >> 4) & km1) << 4);
    aux[3] = ((a1 >> 4) & km2) | (((a2 >> 6) & km1) << 4);
    aux[0] = (a0 & km2) | (((a2 >> 0) & km1) << 4);
    aux[1] = (a1 & km2) | (((a2 >> 2) & km1) << 4);
    thread char* p = (thread char*)aux;
    for (int i = 0; i < 16; ++i) sc[i] = p[i];
}

struct GemvDims { int m; int n_out; int n_in; };

// Each SIMD-group handles one (output row o, token tok). lane = index in simdgroup.
#define GEMV_PROLOGUE \
    uint lane = tid_sg; \
    int o = int(tg.x) * int(sgs_per_tg) + int(sg); \
    int tok = int(tg.y); \
    if (o >= d.n_out || tok >= d.m) return; \
    device const half* xt = x + (size_t)tok * d.n_in;

kernel void gemv_q4_K(device const uchar* W [[buffer(0)]], device const half* x [[buffer(1)]],
                      device half* y [[buffer(2)]], constant GemvDims& d [[buffer(3)]],
                      uint2 tg [[threadgroup_position_in_grid]], uint tid_sg [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]], uint sgs_per_tg [[simdgroups_per_threadgroup]]) {
    GEMV_PROLOGUE
    int nsb = d.n_in / 256;
    device const uchar* wrow = W + (size_t)o * nsb * 144;
    float acc = 0.0f;
    for (int sb = 0; sb < nsb; ++sb) {
        device const uchar* blk = wrow + (size_t)sb * 144;
        float dd = float(*(device const half*)blk), dmin = float(*(device const half*)(blk + 2));
        device const uchar* scales = blk + 4;
        device const uchar* qs = blk + 16;
        device const half* xb = xt + (size_t)sb * 256;
        for (int jj = 0; jj < 4; ++jj) {
            uchar sc, mn;
            get_scale_min_k4(jj*2+0, scales, sc, mn); float d1 = dd*sc, m1 = dmin*mn;
            get_scale_min_k4(jj*2+1, scales, sc, mn); float d2 = dd*sc, m2 = dmin*mn;
            uchar q = qs[jj*32 + lane];
            acc += (d1*float(q & 0xF) - m1) * float(xb[jj*64 + lane]);
            acc += (d2*float(q >> 4)  - m2) * float(xb[jj*64 + 32 + lane]);
        }
    }
    acc = simd_sum(acc);
    if (lane == 0) y[(size_t)tok * d.n_out + o] = half(acc);
}

kernel void gemv_q5_K(device const uchar* W [[buffer(0)]], device const half* x [[buffer(1)]],
                      device half* y [[buffer(2)]], constant GemvDims& d [[buffer(3)]],
                      uint2 tg [[threadgroup_position_in_grid]], uint tid_sg [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]], uint sgs_per_tg [[simdgroups_per_threadgroup]]) {
    GEMV_PROLOGUE
    int nsb = d.n_in / 256;
    device const uchar* wrow = W + (size_t)o * nsb * 176;
    float acc = 0.0f;
    for (int sb = 0; sb < nsb; ++sb) {
        device const uchar* blk = wrow + (size_t)sb * 176;
        float dd = float(*(device const half*)blk), dmin = float(*(device const half*)(blk + 2));
        device const uchar* scales = blk + 4; device const uchar* qh = blk + 16; device const uchar* ql = blk + 48;
        device const half* xb = xt + (size_t)sb * 256;
        uchar qhl = qh[lane];
        for (int jj = 0; jj < 4; ++jj) {
            uchar sc, mn;
            get_scale_min_k4(jj*2+0, scales, sc, mn); float d1 = dd*sc, m1 = dmin*mn;
            get_scale_min_k4(jj*2+1, scales, sc, mn); float d2 = dd*sc, m2 = dmin*mn;
            uchar q = ql[jj*32 + lane];
            acc += (d1*float((q & 0xF) + ((qhl & (1 << (2*jj))) ? 16 : 0)) - m1) * float(xb[jj*64 + lane]);
            acc += (d2*float((q >> 4)  + ((qhl & (2 << (2*jj))) ? 16 : 0)) - m2) * float(xb[jj*64 + 32 + lane]);
        }
    }
    acc = simd_sum(acc);
    if (lane == 0) y[(size_t)tok * d.n_out + o] = half(acc);
}

kernel void gemv_q6_K(device const uchar* W [[buffer(0)]], device const half* x [[buffer(1)]],
                      device half* y [[buffer(2)]], constant GemvDims& d [[buffer(3)]],
                      uint2 tg [[threadgroup_position_in_grid]], uint tid_sg [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]], uint sgs_per_tg [[simdgroups_per_threadgroup]]) {
    GEMV_PROLOGUE
    int nsb = d.n_in / 256;
    device const uchar* wrow = W + (size_t)o * nsb * 210; int is = int(lane) / 16;
    float acc = 0.0f;
    for (int sb = 0; sb < nsb; ++sb) {
        device const uchar* blk = wrow + (size_t)sb * 210;
        device const uchar* ql = blk; device const uchar* qh = blk + 128; device const char* sc = (device const char*)(blk + 192);
        float dd = float(*(device const half*)(blk + 208));
        device const half* xb = xt + (size_t)sb * 256;
        for (int nn = 0; nn < 256; nn += 128) {
            device const uchar* Ql = ql + (nn/128)*64; device const uchar* Qh = qh + (nn/128)*32; device const char* Sc = sc + (nn/128)*8;
            uchar ql0 = Ql[lane], ql1 = Ql[lane+32], qhb = Qh[lane];
            int q1 = int((ql0 & 0xF) | (((qhb >> 0) & 3) << 4)) - 32;
            int q2 = int((ql1 & 0xF) | (((qhb >> 2) & 3) << 4)) - 32;
            int q3 = int((ql0 >> 4)  | (((qhb >> 4) & 3) << 4)) - 32;
            int q4 = int((ql1 >> 4)  | (((qhb >> 6) & 3) << 4)) - 32;
            acc += dd*Sc[is+0]*q1 * float(xb[nn+lane+0]);
            acc += dd*Sc[is+2]*q2 * float(xb[nn+lane+32]);
            acc += dd*Sc[is+4]*q3 * float(xb[nn+lane+64]);
            acc += dd*Sc[is+6]*q4 * float(xb[nn+lane+96]);
        }
    }
    acc = simd_sum(acc);
    if (lane == 0) y[(size_t)tok * d.n_out + o] = half(acc);
}

kernel void gemv_q3_K(device const uchar* W [[buffer(0)]], device const half* x [[buffer(1)]],
                      device half* y [[buffer(2)]], constant GemvDims& d [[buffer(3)]],
                      uint2 tg [[threadgroup_position_in_grid]], uint tid_sg [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]], uint sgs_per_tg [[simdgroups_per_threadgroup]]) {
    GEMV_PROLOGUE
    int nsb = d.n_in / 256;
    device const uchar* wrow = W + (size_t)o * nsb * 110;
    float acc = 0.0f;
    for (int sb = 0; sb < nsb; ++sb) {
        device const uchar* blk = wrow + (size_t)sb * 110; device const uchar* hmask = blk; device const uchar* qs = blk + 32;
        float dd = float(*(device const half*)(blk + 108)); char scl[16]; unpack_q3_scales(blk + 96, scl);
        device const half* xb = xt + (size_t)sb * 256; uchar hml = hmask[lane];
        for (int h = 0; h < 2; ++h) {
            uchar qb = qs[h*32 + lane];
            for (int j = 0; j < 4; ++j) {
                uchar mbit = uchar(1u << (h*4 + j));
                int sidx = h*8 + j*2 + (int(lane) >= 16 ? 1 : 0);
                int ql = (qb >> (2*j)) & 3; int hb = (hml & mbit) ? 0 : 4;
                acc += dd*float(scl[sidx] - 32)*float(ql - hb) * float(xb[h*128 + j*32 + lane]);
            }
        }
    }
    acc = simd_sum(acc);
    if (lane == 0) y[(size_t)tok * d.n_out + o] = half(acc);
}

kernel void gemv_q2_K(device const uchar* W [[buffer(0)]], device const half* x [[buffer(1)]],
                      device half* y [[buffer(2)]], constant GemvDims& d [[buffer(3)]],
                      uint2 tg [[threadgroup_position_in_grid]], uint tid_sg [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]], uint sgs_per_tg [[simdgroups_per_threadgroup]]) {
    GEMV_PROLOGUE
    int nsb = d.n_in / 256;
    device const uchar* wrow = W + (size_t)o * nsb * 84;
    float acc = 0.0f;
    for (int sb = 0; sb < nsb; ++sb) {
        device const uchar* blk = wrow + (size_t)sb * 84; device const uchar* scales = blk; device const uchar* qs = blk + 16;
        float dd = float(*(device const half*)(blk + 80)), dmin = float(*(device const half*)(blk + 82));
        device const half* xb = xt + (size_t)sb * 256;
        for (int h = 0; h < 2; ++h) {
            uchar qb = qs[h*32 + lane];
            for (int j = 0; j < 4; ++j) {
                uchar sc = scales[h*8 + j*2 + (int(lane) >= 16 ? 1 : 0)];
                float dl = dd*float(sc & 0xF), ml = dmin*float(sc >> 4);
                acc += (dl*float((qb >> (2*j)) & 3) - ml) * float(xb[h*128 + j*32 + lane]);
            }
        }
    }
    acc = simd_sum(acc);
    if (lane == 0) y[(size_t)tok * d.n_out + o] = half(acc);
}

// --- forward ops ---
struct NormArgs { int n_rows; int dim; float eps; };
// One SIMD-group per row: sum of squares -> rms -> scale by weight.
kernel void rmsnorm(device const half* x [[buffer(0)]], device const half* w [[buffer(1)]],
                    device half* out [[buffer(2)]], constant NormArgs& a [[buffer(3)]],
                    uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    if (int(row) >= a.n_rows) return;
    device const half* xr = x + (size_t)row * a.dim; device half* orow = out + (size_t)row * a.dim;
    float ss = 0.0f;
    for (int i = int(lane); i < a.dim; i += WARP) { float v = float(xr[i]); ss += v*v; }
    ss = simd_sum(ss);
    float inv = rsqrt(ss / float(a.dim) + a.eps);
    for (int i = int(lane); i < a.dim; i += WARP) orow[i] = half(float(xr[i]) * inv * float(w[i]));
}

struct RopeArgs { int n_tokens; int n_heads; int head_dim; float base; };
kernel void rope(device half* x [[buffer(0)]], device const int* pos [[buffer(1)]],
                 constant RopeArgs& a [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    int half_hd = a.head_dim / 2;
    int total = a.n_tokens * a.n_heads * half_hd;
    if (int(gid) >= total) return;
    int i = int(gid) % half_hd; int hh = int(gid) / half_hd; int h = hh % a.n_heads; int t = hh / a.n_heads;
    float freq = pow(a.base, -2.0f * float(i) / float(a.head_dim));
    float ang = float(pos[t]) * freq; float c = cos(ang), s = sin(ang);
    device half* xr = x + ((size_t)t * a.n_heads + h) * a.head_dim;
    float x0 = float(xr[i]), x1 = float(xr[i + half_hd]);
    xr[i] = half(x0*c - x1*s); xr[i + half_hd] = half(x0*s + x1*c);
}

kernel void silu_mul(device half* gate [[buffer(0)]], device const half* up [[buffer(1)]],
                     constant int& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (int(i) >= n) return;
    float g = float(gate[i]); gate[i] = half((g / (1.0f + exp(-g))) * float(up[i]));
}
kernel void residual_add(device half* x [[buffer(0)]], device const half* y [[buffer(1)]],
                         constant int& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (int(i) >= n) return;
    x[i] = half(float(x[i]) + float(y[i]));
}
kernel void embed(device const half* table [[buffer(0)]], device const int* ids [[buffer(1)]],
                  device half* out [[buffer(2)]], constant int2& nd [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    int n = nd.x, dim = nd.y; if (int(gid) >= n*dim) return;
    int t = int(gid) / dim, c = int(gid) % dim;
    out[gid] = table[(size_t)ids[t]*dim + c];
}

// Naive causal attention: one SIMD-group per (query token, head). Online-ish two pass over keys.
struct AttnArgs { int n_new; int len_before; int n_heads; int n_kv_heads; int head_dim; float scale; };
kernel void attention(device const half* Q [[buffer(0)]], device const half* Kc [[buffer(1)]],
                      device const half* Vc [[buffer(2)]], device half* out [[buffer(3)]],
                      constant AttnArgs& a [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                      uint sgs [[simdgroups_per_threadgroup]]) {
    int t = int(tg.x); int h = int(tg.y) * int(sgs) + int(sg);
    if (t >= a.n_new || h >= a.n_heads) return;
    int hd = a.head_dim, kvd = a.n_kv_heads * hd;
    int kvh = h / (a.n_heads / a.n_kv_heads);
    int qpos = a.len_before + t; int nkeys = qpos + 1;
    device const half* q = Q + ((size_t)t * a.n_heads + h) * hd;
    // pass 1: max score
    float mx = -1e30f;
    for (int k = int(lane); k < nkeys; k += WARP) {
        device const half* kk = Kc + (size_t)k * kvd + kvh * hd;
        float s = 0.0f; for (int i = 0; i < hd; ++i) s += float(q[i]) * float(kk[i]);
        mx = max(mx, s * a.scale);
    }
    mx = simd_max(mx);
    // pass 2: sum exp + weighted V (each lane accumulates its key subset, then reduce)
    float denom = 0.0f;
    for (int i = int(lane); i < hd; i += WARP) {  // clear output slots this lane owns
        // handled below after computing acc
    }
    // accumulate per-lane into threadgroup-free local: recompute over keys, add to out via simd sums per dim
    for (int i = 0; i < hd; ++i) {
        float acc = 0.0f;
        for (int k = int(lane); k < nkeys; k += WARP) {
            device const half* kk = Kc + (size_t)k * kvd + kvh * hd;
            device const half* vv = Vc + (size_t)k * kvd + kvh * hd;
            float s = 0.0f; for (int j = 0; j < hd; ++j) s += float(q[j]) * float(kk[j]);
            float w = exp(s * a.scale - mx);
            acc += w * float(vv[i]);
            if (i == 0) denom += w;
        }
        acc = simd_sum(acc);
        if (i == 0) denom = simd_sum(denom);
        if (lane == 0) out[((size_t)t * a.n_heads + h) * hd + i] = half(acc / denom);
    }
}

// One threadgroup reduces the whole vocab to an argmax token id.
kernel void argmax(device const half* logits [[buffer(0)]], constant int& n [[buffer(1)]],
                   device int* out [[buffer(2)]], uint lane [[thread_index_in_simdgroup]],
                   uint sg [[simdgroup_index_in_threadgroup]], uint sgs [[simdgroups_per_threadgroup]],
                   threadgroup float* sval [[threadgroup(0)]], threadgroup int* sidx [[threadgroup(1)]]) {
    int tid = int(sg) * WARP + int(lane);
    int nthreads = int(sgs) * WARP;
    float best = -1e30f; int bi = 0;
    for (int v = tid; v < n; v += nthreads) { float f = float(logits[v]); if (f > best) { best = f; bi = v; } }
    // simd reduce within group, then across groups via threadgroup memory
    float gmax = simd_max(best);
    if (best == gmax) { sval[sg] = best; sidx[sg] = bi; }   // one winner per simdgroup
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float bv = sval[0]; int bx = sidx[0];
        for (uint g = 1; g < sgs; ++g) if (sval[g] > bv) { bv = sval[g]; bx = sidx[g]; }
        out[0] = bx;
    }
}

// Standalone Metal-backend smoke test: fused Q4_K GEMV on the GPU vs a CPU reference.
// Proves the MSL kernel + host glue compute correctly on Apple Silicon. Run with
//   FLOATY_METALLIB=<path>/floaty_kernels.metallib ./test_metal
#include "../backend.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstring>
#include <Foundation/Foundation.h>
#include <simd/simd.h>

static float h2f(uint16_t h) {  // minimal fp16 -> fp32
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1F, m = h & 0x3FF, out;
    if (e == 0) { if (m == 0) out = s << 31; else { e = 127 - 15 + 1; while (!(m & 0x400)) { m <<= 1; e--; } m &= 0x3FF; out = (s << 31) | (e << 23) | (m << 13); } }
    else if (e == 31) out = (s << 31) | 0x7F800000 | (m << 13);
    else out = (s << 31) | ((e - 15 + 127) << 23) | (m << 13);
    float f; memcpy(&f, &out, 4); return f;
}
static uint16_t f2h(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t s = (x >> 16) & 0x8000; int e = ((x >> 23) & 0xFF) - 127 + 15; uint32_t m = x & 0x7FFFFF;
    if (e <= 0) return s;
    if (e >= 31) return s | 0x7C00;
    return s | (e << 10) | (m >> 13);
}
static void get_scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
    else { d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

int main() {
    Backend* b = make_metal_backend();
    if (!b) { printf("FAIL: no metal backend\n"); return 1; }
    printf("backend: %s\n", b->name());

    int n_out = 8, n_in = 256, m = 2, nsb = n_in / 256;     // 1 super-block/row
    size_t wbytes = (size_t)n_out * nsb * 144;
    std::vector<uint8_t> W(wbytes); for (auto& x : W) x = rand() & 0xFF;
    for (int o = 0; o < n_out; ++o) {
        uint8_t* blk = W.data() + (size_t)o * 144;
        *(uint16_t*)(blk + 0) = f2h(0.05f * ((o % 5) + 1));   // d
        *(uint16_t*)(blk + 2) = f2h(0.03f * ((o % 3) + 1));   // dmin
    }
    std::vector<uint16_t> x(m * n_in), y(m * n_out, 0);
    for (auto& v : x) v = f2h((rand() / (float)RAND_MAX) * 2 - 1);

    // CPU reference: dequant each row, dot with x.
    std::vector<float> yref(m * n_out, 0);
    for (int o = 0; o < n_out; ++o) {
        const uint8_t* blk = W.data() + (size_t)o * 144;
        float d = h2f(*(uint16_t*)blk), dmin = h2f(*(uint16_t*)(blk + 2));
        const uint8_t* sc = blk + 4; const uint8_t* qs = blk + 16;
        float row[256]; int yi = 0;
        for (int g = 0; g < 256; g += 64) {
            uint8_t s1, m1b, s2, m2b;
            get_scale_min_k4(g / 32 + 0, sc, s1, m1b); float d1 = d * s1, mn1 = dmin * m1b;
            get_scale_min_k4(g / 32 + 1, sc, s2, m2b); float d2 = d * s2, mn2 = dmin * m2b;
            const uint8_t* q = qs + (g / 64) * 32;
            for (int l = 0; l < 32; ++l) row[yi++] = d1 * (q[l] & 0xF) - mn1;
            for (int l = 0; l < 32; ++l) row[yi++] = d2 * (q[l] >> 4)  - mn2;
        }
        for (int t = 0; t < m; ++t) { float s = 0; for (int i = 0; i < n_in; ++i) s += row[i] * h2f(x[t * n_in + i]); yref[t * n_out + o] = s; }
    }

    void* dW = b->alloc(wbytes); b->upload(dW, W.data(), wbytes);
    void* dx = b->alloc(x.size() * 2); b->upload(dx, x.data(), x.size() * 2);
    void* dy = b->alloc(y.size() * 2);
    b->fused_gemv(dW, dx, dy, m, n_out, n_in, 12);   // Q4_K
    b->sync();
    b->download(y.data(), dy, y.size() * 2);

    float mr = 0;
    for (int i = 0; i < m * n_out; ++i) { float e = fabsf(h2f(y[i]) - yref[i]) / (fabsf(yref[i]) + 1e-2f); mr = fmaxf(mr, e); }
    printf("Q4_K gemv: max rel err = %.4f  %s\n", mr, mr < 0.03f ? "PASS" : "FAIL");
    return mr < 0.03f ? 0 : 1;
}

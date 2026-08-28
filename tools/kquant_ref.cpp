// K-quant dequant self-test (CPU, no GPU). Hand-builds Q4_K/Q6_K blocks with
// known outputs. Include impl to compile standalone. Single-line comments only.
#include "../src/kquant.cpp"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>

static int g_fail = 0;
static void check(bool c, const char* what) {
    printf(c ? "[PASS] %s\n" : "[FAIL] %s\n", what);
    if (!c) ++g_fail;
}
static bool approx(float a, float b) { return fabsf(a - b) < 1e-3f; }

int main() {
    printf("[INFO] K-quant dequant test\n");

    // --- Q6_K: ql=0, qh=0, scales=1, d=1.0 -> every q = (0)-32 = -32; y = 1*1*(-32).
    {
        std::vector<uint8_t> blk(210, 0);
        for (int i = 192; i < 208; ++i) blk[i] = 1;      // scales[16] = 1 (int8)
        blk[208] = 0x00; blk[209] = 0x3C;                // d = fp16 1.0
        std::vector<float> y(256);
        dequant_q6_K_f32(blk.data(), 256, y.data());
        bool ok = true;
        for (int i = 0; i < 256; ++i) if (!approx(y[i], -32.0f)) ok = false;
        check(ok, "Q6_K all-zero quants, scale=1, d=1 -> all -32");
        printf("[INFO]   y[0]=%.1f y[128]=%.1f\n", y[0], y[128]);
    }

    // --- Q4_K: d=1, dmin=0, scales=1, qs=0x21 -> low nibble 1, high nibble 2.
    {
        std::vector<uint8_t> blk(144, 0);
        blk[0] = 0x00; blk[1] = 0x3C;                    // d = 1.0
        blk[2] = 0x00; blk[3] = 0x00;                    // dmin = 0
        for (int i = 4; i < 16; ++i) blk[i] = 1;         // scales[12] = 1
        for (int i = 16; i < 144; ++i) blk[i] = 0x21;    // qs = 0x21 (lo=1, hi=2)
        std::vector<float> y(256);
        dequant_q4_K_f32(blk.data(), 256, y.data());
        // each 64-group: 32 low (=1) then 32 high (=2), repeated 4x
        bool ok = true;
        for (int g = 0; g < 4; ++g) {
            for (int l = 0; l < 32; ++l) if (!approx(y[g*64 + l], 1.0f)) ok = false;
            for (int l = 0; l < 32; ++l) if (!approx(y[g*64 + 32 + l], 2.0f)) ok = false;
        }
        check(ok, "Q4_K d=1 dmin=0 scale=1 qs=0x21 -> [1x32, 2x32] x4");
        printf("[INFO]   y[0]=%.1f y[32]=%.1f y[64]=%.1f\n", y[0], y[32], y[64]);
    }

    // --- Q6_K nonzero: set ql low nibble to 0xF, qh bits 0, scale=1, d=1 ->
    //     q1 = (15 | 0) - 32 = -17.
    {
        std::vector<uint8_t> blk(210, 0);
        for (int i = 0; i < 128; ++i) blk[i] = 0x0F;     // ql all 0x0F
        for (int i = 192; i < 208; ++i) blk[i] = 1;
        blk[208] = 0x00; blk[209] = 0x3C;
        std::vector<float> y(256);
        dequant_q6_K_f32(blk.data(), 256, y.data());
        // y[0] uses q1 = (ql[0]&0xF | ...) - 32 = 15 - 32 = -17
        check(approx(y[0], -17.0f), "Q6_K ql=0x0F -> q1 = 15-32 = -17");
        printf("[INFO]   y[0]=%.1f\n", y[0]);
    }

    if (g_fail == 0) { printf("[INFO] ALL CHECKS PASSED\n"); return 0; }
    printf("[INFO] %d CHECK(S) FAILED\n", g_fail);
    return 1;
}

// Q8_0 dequant reference + self-test (CPU, no GPU needed).
// Mirrors src/dequant.cu (x = decode_fp16(d) * q) so the algorithm and the exact
// 34-byte GGUF block layout can be validated locally. Doubles as the host golden
// reference for the CUDA unit test. Uses only single-line comments (linter rule).
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdarg>
#include <vector>

static constexpr int QK8_0 = 32;

// GGUF block_q8_0 byte layout: fp16 scale (raw bits) then 32 int8 quants = 34 bytes.
#pragma pack(push, 1)
struct BlockQ80 {
    uint16_t d;          // fp16 scale, raw bits (as stored in GGUF)
    int8_t   qs[QK8_0];  // quantized values
};
#pragma pack(pop)
static_assert(sizeof(BlockQ80) == 34, "BlockQ80 must be 34 bytes");

// IEEE-754 half -> float.
static float half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            exp = 127 - 15 + 1;
            while ((mant & 0x400u) == 0) { mant <<= 1; exp--; }
            mant &= 0x3FFu;
            f = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &f, 4);
    return out;
}

// IEEE-754 float -> half (round toward nearest; good enough for test scales).
static uint16_t float_to_half(float x) {
    uint32_t f;
    memcpy(&f, &x, 4);
    uint32_t sign = (f >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((f >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = f & 0x7FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        int shift = 14 - exp;
        uint32_t m = mant >> shift;
        if ((mant >> (shift - 1)) & 1u) m += 1;
        return (uint16_t)(sign | m);
    } else if (exp >= 0x1F) {
        return (uint16_t)(sign | 0x7C00u);
    }
    uint16_t h = (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
    if (mant & 0x1000u) h += 1;
    return h;
}

// The algorithm under test: dequant Q8_0 -> float. Mirror of dequant_q8_0_kernel.
static void dequant_ref(const BlockQ80* blocks, float* out, int n_blocks) {
    for (int b = 0; b < n_blocks; ++b) {
        float d = half_to_float(blocks[b].d);
        for (int j = 0; j < QK8_0; ++j) {
            out[b * QK8_0 + j] = d * (float)blocks[b].qs[j];
        }
    }
}

// Minimal leveled logger.
static int g_fail = 0;
static void logline(const char* level, const char* fmt, ...) {
    printf("[%-4s] ", level);
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    printf("\n");
}
static void check(bool cond, const char* what) {
    if (cond) {
        logline("PASS", "%s", what);
    } else {
        logline("FAIL", "%s", what);
        ++g_fail;
    }
}

int main() {
    logline("INFO", "Q8_0 dequant algorithm test");
    logline("INFO", "sizeof(BlockQ80) = %zu bytes (expect 34)", sizeof(BlockQ80));
    check(sizeof(BlockQ80) == 34, "block layout is 34 bytes");

    std::vector<float> out(QK8_0);

    // Case A: scale = 1.0 -> output must equal q exactly (independent ground truth).
    BlockQ80 a;
    a.d = float_to_half(1.0f);
    for (int j = 0; j < QK8_0; ++j) a.qs[j] = (int8_t)(j - 16);
    dequant_ref(&a, out.data(), 1);
    bool a_ok = true;
    for (int j = 0; j < QK8_0; ++j) if (out[j] != (float)a.qs[j]) a_ok = false;
    check(a_ok, "scale=1.0 -> out == q for all 32 values");
    logline("INFO", "  sample: q[0]=%d -> %.4f | q[31]=%d -> %.4f",
            a.qs[0], out[0], a.qs[31], out[31]);

    // Case B: scale = 0.5 -> output must equal 0.5*q exactly (0.5 is fp16-exact).
    BlockQ80 b;
    b.d = float_to_half(0.5f);
    for (int j = 0; j < QK8_0; ++j) b.qs[j] = (int8_t)(2 * (j - 16));
    dequant_ref(&b, out.data(), 1);
    bool b_ok = true;
    for (int j = 0; j < QK8_0; ++j) if (out[j] != 0.5f * (float)b.qs[j]) b_ok = false;
    check(b_ok, "scale=0.5 -> out == 0.5*q for all 32 values");
    logline("INFO", "  sample: q[0]=%d -> %.4f | q[20]=%d -> %.4f",
            b.qs[0], out[0], b.qs[20], out[20]);

    // Case C: boundary quants with a small scale (min/max int8, zero).
    BlockQ80 c;
    float cs = 0.01f;
    c.d = float_to_half(cs);
    c.qs[0] = -128; c.qs[1] = 127; c.qs[2] = 0; c.qs[3] = 1;
    for (int j = 4; j < QK8_0; ++j) c.qs[j] = 0;
    dequant_ref(&c, out.data(), 1);
    float ds = half_to_float(c.d);
    bool c_ok = out[0] == ds * -128.0f && out[1] == ds * 127.0f &&
                out[2] == 0.0f && out[3] == ds * 1.0f;
    check(c_ok, "boundary quants (-128,127,0,1) scale correctly");
    logline("INFO", "  scale=%.6f (fp16) | -128 -> %.4f | 127 -> %.4f | 0 -> %.4f",
            ds, out[0], out[1], out[2]);

    if (g_fail == 0) {
        logline("INFO", "ALL CHECKS PASSED");
        return 0;
    }
    logline("INFO", "%d CHECK(S) FAILED", g_fail);
    return 1;
}

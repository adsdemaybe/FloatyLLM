// RoPE reference + self-test (CPU). Mirrors src/rope.cu (interleaved pairs).
// Validates the algorithm locally; single-line comments only (linter rule).
#include <cstdio>
#include <cmath>
#include <cstdarg>
#include <vector>

// The algorithm under test: interleaved RoPE, in-place.
static void rope_ref(float* x, const int* pos, int n_tokens, int n_heads,
                     int head_dim, float base) {
    int half = head_dim / 2;
    for (int blk = 0; blk < n_tokens * n_heads; ++blk) {
        int token = blk / n_heads;
        int p = pos[token];
        float* v = x + (size_t)blk * head_dim;
        for (int j = 0; j < half; ++j) {
            float freq = powf(base, -2.0f * (float)j / (float)head_dim);
            float ang = (float)p * freq;
            float c = cosf(ang), s = sinf(ang);
            float a = v[2 * j], b = v[2 * j + 1];
            v[2 * j]     = a * c - b * s;
            v[2 * j + 1] = a * s + b * c;
        }
    }
}

static int g_fail = 0;
static void logline(const char* level, const char* fmt, ...) {
    printf("[%-4s] ", level);
    va_list ap; va_start(ap, fmt); vprintf(fmt, ap); va_end(ap);
    printf("\n");
}
static void check(bool cond, const char* what) {
    if (cond) logline("PASS", "%s", what);
    else { logline("FAIL", "%s", what); ++g_fail; }
}
static bool approx(float a, float b, float tol) { return fabsf(a - b) <= tol; }

int main() {
    const float base = 10000.0f;
    logline("INFO", "RoPE algorithm test");

    // Case A: pos=0 -> angle 0 -> identity.
    {
        int hd = 8;
        std::vector<float> x(hd), orig(hd);
        for (int j = 0; j < hd; ++j) x[j] = orig[j] = 0.1f * (float)(j + 1);
        int pos = 0;
        rope_ref(x.data(), &pos, 1, 1, hd, base);
        bool ok = true;
        for (int j = 0; j < hd; ++j) if (!approx(x[j], orig[j], 1e-6f)) ok = false;
        check(ok, "pos=0 leaves the vector unchanged");
    }

    // Case B: rotation preserves each pair's norm.
    {
        int hd = 16;
        std::vector<float> x(hd), orig(hd);
        for (int j = 0; j < hd; ++j) x[j] = orig[j] = 0.3f * (float)(j - 8) + 0.5f;
        int pos = 7;
        rope_ref(x.data(), &pos, 1, 1, hd, base);
        bool ok = true;
        for (int j = 0; j < hd / 2; ++j) {
            float n0 = orig[2*j]*orig[2*j] + orig[2*j+1]*orig[2*j+1];
            float n1 = x[2*j]*x[2*j] + x[2*j+1]*x[2*j+1];
            if (!approx(n0, n1, 1e-4f)) ok = false;
        }
        check(ok, "rotation preserves per-pair norm");
    }

    // Case C: known numeric spot value. head_dim=4, pos=1, x=[1,0,1,0].
    // j=0 angle=1 rad -> (cos1, sin1); j=1 angle=0.01 rad -> (cos.01, sin.01).
    {
        int hd = 4;
        std::vector<float> x = {1.0f, 0.0f, 1.0f, 0.0f};
        int pos = 1;
        rope_ref(x.data(), &pos, 1, 1, hd, base);
        bool ok = approx(x[0], cosf(1.0f), 1e-5f) && approx(x[1], sinf(1.0f), 1e-5f) &&
                  approx(x[2], cosf(0.01f), 1e-5f) && approx(x[3], sinf(0.01f), 1e-5f);
        check(ok, "spot value matches manual cos/sin rotation");
        logline("INFO", "  pair0=(%.4f,%.4f) expect(%.4f,%.4f)",
                x[0], x[1], cosf(1.0f), sinf(1.0f));
    }

    if (g_fail == 0) { logline("INFO", "ALL CHECKS PASSED"); return 0; }
    logline("INFO", "%d CHECK(S) FAILED", g_fail);
    return 1;
}

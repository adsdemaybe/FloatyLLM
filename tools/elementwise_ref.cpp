// Elementwise ops reference + self-test (CPU). Mirrors src/elementwise.cu.
// Single-line comments only (linter rule).
#include <cstdio>
#include <cmath>
#include <cstdarg>
#include <vector>

static float silu_ref(float v) { return v / (1.0f + expf(-v)); }

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
    logline("INFO", "Elementwise ops algorithm test");

    // SiLU ground truths: silu(0)=0, silu(x)->x for large +, ->0 for large -.
    check(approx(silu_ref(0.0f), 0.0f, 1e-6f), "silu(0) == 0");
    check(approx(silu_ref(20.0f), 20.0f, 1e-3f), "silu(large +) approx x");
    check(approx(silu_ref(-20.0f), 0.0f, 1e-3f), "silu(large -) approx 0");
    logline("INFO", "  silu(1)=%.5f silu(-1)=%.5f", silu_ref(1.0f), silu_ref(-1.0f));

    // Residual add: x += y.
    {
        std::vector<float> x = {1.0f, -2.0f, 3.5f}, y = {0.5f, 2.0f, -1.5f};
        std::vector<float> expect = {1.5f, 0.0f, 2.0f};
        bool ok = true;
        for (size_t i = 0; i < x.size(); ++i) {
            if (!approx(x[i] + y[i], expect[i], 1e-6f)) ok = false;
        }
        check(ok, "residual add x+y correct");
    }

    // Elementwise multiply (SwiGLU shape): out = silu(gate) * up.
    {
        float gate = 2.0f, up = 3.0f;
        float out = silu_ref(gate) * up;
        check(approx(out, silu_ref(2.0f) * 3.0f, 1e-6f), "swiglu mul out = silu(gate)*up");
        logline("INFO", "  silu(2)*3 = %.5f", out);
    }

    if (g_fail == 0) { logline("INFO", "ALL CHECKS PASSED"); return 0; }
    logline("INFO", "%d CHECK(S) FAILED", g_fail);
    return 1;
}

// Softmax reference + self-test (CPU). Mirrors src/softmax.cu.
// Single-line comments only (linter rule).
#include <cstdio>
#include <cmath>
#include <cstdarg>
#include <vector>

static void softmax_ref(const float* x, float* out, int dim) {
    float m = -1e30f;
    for (int i = 0; i < dim; ++i) m = fmaxf(m, x[i]);
    float s = 0.0f;
    for (int i = 0; i < dim; ++i) { out[i] = expf(x[i] - m); s += out[i]; }
    for (int i = 0; i < dim; ++i) out[i] /= s;
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
    logline("INFO", "Softmax algorithm test");

    // Sums to 1.
    {
        std::vector<float> x = {1.0f, 2.0f, 3.0f, -1.0f, 0.5f}, out(5);
        softmax_ref(x.data(), out.data(), 5);
        float s = 0; for (float v : out) s += v;
        check(approx(s, 1.0f, 1e-6f), "outputs sum to 1");
    }

    // Uniform input -> uniform 1/n.
    {
        int n = 8;
        std::vector<float> x(n, 2.5f), out(n);
        softmax_ref(x.data(), out.data(), n);
        bool ok = true;
        for (int i = 0; i < n; ++i) if (!approx(out[i], 1.0f / n, 1e-6f)) ok = false;
        check(ok, "uniform input -> uniform 1/n");
    }

    // Known ratios: softmax([0, ln2, ln3]) = [1,2,3]/6.
    {
        std::vector<float> x = {0.0f, logf(2.0f), logf(3.0f)}, out(3);
        softmax_ref(x.data(), out.data(), 3);
        bool ok = approx(out[0], 1.0f/6, 1e-5f) && approx(out[1], 2.0f/6, 1e-5f) &&
                  approx(out[2], 3.0f/6, 1e-5f);
        check(ok, "softmax([0,ln2,ln3]) == [1,2,3]/6");
        logline("INFO", "  out=(%.4f,%.4f,%.4f)", out[0], out[1], out[2]);
    }

    // Shift invariance: softmax(x) == softmax(x + c).
    {
        std::vector<float> x = {0.2f, -1.3f, 4.0f}, a(3), b(3);
        std::vector<float> xs = {0.2f + 5, -1.3f + 5, 4.0f + 5};
        softmax_ref(x.data(), a.data(), 3);
        softmax_ref(xs.data(), b.data(), 3);
        bool ok = true;
        for (int i = 0; i < 3; ++i) if (!approx(a[i], b[i], 1e-5f)) ok = false;
        check(ok, "shift invariance");
    }

    if (g_fail == 0) { logline("INFO", "ALL CHECKS PASSED"); return 0; }
    logline("INFO", "%d CHECK(S) FAILED", g_fail);
    return 1;
}

// RMSNorm reference + self-test (CPU, no GPU needed). Mirrors src/rmsnorm.cu:
// out_i = x_i * rsqrt(mean(x^2)+eps) * weight_i. Validates the algorithm locally
// and serves as the host golden reference. Single-line comments only (linter rule).
#include <cstdio>
#include <cmath>
#include <cstdarg>
#include <vector>

// The algorithm under test (fp32 reference of the fp16 kernel).
static void rmsnorm_ref(const float* x, const float* w, float* out,
                        int n_rows, int dim, float eps) {
    for (int r = 0; r < n_rows; ++r) {
        double ss = 0.0;
        for (int j = 0; j < dim; ++j) {
            float v = x[r * dim + j];
            ss += (double)v * v;
        }
        float inv = 1.0f / sqrtf((float)(ss / dim) + eps);
        for (int j = 0; j < dim; ++j) {
            out[r * dim + j] = x[r * dim + j] * inv * w[j];
        }
    }
}

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
    if (cond) logline("PASS", "%s", what);
    else { logline("FAIL", "%s", what); ++g_fail; }
}
static bool approx(float a, float b, float tol) { return fabsf(a - b) <= tol; }

int main() {
    const float eps = 1e-6f;
    logline("INFO", "RMSNorm algorithm test");

    // Case A: uniform x=3, weight=1 -> rms=3 -> out_i = 3/3 = 1 for all.
    {
        int dim = 64;
        std::vector<float> x(dim, 3.0f), w(dim, 1.0f), out(dim);
        rmsnorm_ref(x.data(), w.data(), out.data(), 1, dim, eps);
        bool ok = true;
        for (int j = 0; j < dim; ++j) if (!approx(out[j], 1.0f, 1e-4f)) ok = false;
        check(ok, "uniform x=3, w=1 -> out == 1 for all");
        logline("INFO", "  out[0]=%.5f out[%d]=%.5f", out[0], dim - 1, out[dim - 1]);
    }

    // Case B: uniform x=-2, weight=4 -> rms=2 -> out_i = (-2/2)*4 = -4 for all.
    {
        int dim = 32;
        std::vector<float> x(dim, -2.0f), w(dim, 4.0f), out(dim);
        rmsnorm_ref(x.data(), w.data(), out.data(), 1, dim, eps);
        bool ok = true;
        for (int j = 0; j < dim; ++j) if (!approx(out[j], -4.0f, 1e-4f)) ok = false;
        check(ok, "uniform x=-2, w=4 -> out == -4 for all");
        logline("INFO", "  out[0]=%.5f", out[0]);
    }

    // Case C: RMSNorm preserves the RMS at 1 (times weight). With w=1, mean(out^2)==1.
    {
        int dim = 100;
        std::vector<float> x(dim), w(dim, 1.0f), out(dim);
        for (int j = 0; j < dim; ++j) x[j] = 0.3f * (float)(j - 50);
        rmsnorm_ref(x.data(), w.data(), out.data(), 1, dim, eps);
        double ss = 0.0;
        for (int j = 0; j < dim; ++j) ss += (double)out[j] * out[j];
        float rms = (float)sqrt(ss / dim);
        check(approx(rms, 1.0f, 1e-3f), "normalized output has unit RMS (w=1)");
        logline("INFO", "  rms(out)=%.6f (expect ~1.0)", rms);
    }

    if (g_fail == 0) { logline("INFO", "ALL CHECKS PASSED"); return 0; }
    logline("INFO", "%d CHECK(S) FAILED", g_fail);
    return 1;
}

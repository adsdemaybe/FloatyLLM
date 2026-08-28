// Causal attention reference + self-test (CPU). Direct softmax attention;
// the GPU kernel uses the mathematically-equivalent online softmax.
// Single-line comments only (linter rule).
#include <cstdio>
#include <cmath>
#include <cstdarg>
#include <vector>

// Q:[T][H][D], K/V:[T][KVH][D], out:[T][H][D]. Causal, GQA-aware.
static void attention_ref(const float* Q, const float* K, const float* V, float* out,
                          int T, int H, int KVH, int D, float scale) {
    int group = H / KVH;
    std::vector<float> scores(T);
    for (int t = 0; t < T; ++t) {
        for (int h = 0; h < H; ++h) {
            int kvh = h / group;
            const float* q = Q + ((size_t)t * H + h) * D;
            float m = -1e30f;
            for (int j = 0; j <= t; ++j) {
                const float* k = K + ((size_t)j * KVH + kvh) * D;
                float s = 0.0f;
                for (int d = 0; d < D; ++d) s += q[d] * k[d];
                scores[j] = s * scale;
                m = fmaxf(m, scores[j]);
            }
            float denom = 0.0f;
            for (int j = 0; j <= t; ++j) { scores[j] = expf(scores[j] - m); denom += scores[j]; }
            float* o = out + ((size_t)t * H + h) * D;
            for (int d = 0; d < D; ++d) o[d] = 0.0f;
            for (int j = 0; j <= t; ++j) {
                const float* v = V + ((size_t)j * KVH + kvh) * D;
                float p = scores[j] / denom;
                for (int d = 0; d < D; ++d) o[d] += p * v[d];
            }
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
    logline("INFO", "Causal attention algorithm test");
    const float scale = 1.0f;

    // Case A: single token attends only to itself -> out == V[0].
    {
        int T = 1, H = 1, KVH = 1, D = 4;
        std::vector<float> Q(D, 0.5f), K(D, 0.3f);
        std::vector<float> V = {1.0f, 2.0f, 3.0f, 4.0f}, out(D);
        attention_ref(Q.data(), K.data(), V.data(), out.data(), T, H, KVH, D, scale);
        bool ok = true;
        for (int d = 0; d < D; ++d) if (!approx(out[d], V[d], 1e-6f)) ok = false;
        check(ok, "single token -> out == V[0]");
    }

    // Case B: Q=0 -> uniform scores -> out_t = causal mean of V[0..t].
    {
        int T = 3, H = 1, KVH = 1, D = 2;
        std::vector<float> Q(T * D, 0.0f);
        std::vector<float> K(T * D, 0.9f);
        std::vector<float> V = {2.0f, 0.0f, 4.0f, 10.0f, 6.0f, 20.0f}, out(T * D);
        attention_ref(Q.data(), K.data(), V.data(), out.data(), T, H, KVH, D, scale);
        // t=0 -> V0=(2,0); t=1 -> mean(V0,V1)=(3,5); t=2 -> mean(V0,V1,V2)=(4,10)
        bool ok = approx(out[0], 2.0f, 1e-5f) && approx(out[1], 0.0f, 1e-5f) &&
                  approx(out[2], 3.0f, 1e-5f) && approx(out[3], 5.0f, 1e-5f) &&
                  approx(out[4], 4.0f, 1e-5f) && approx(out[5], 10.0f, 1e-5f);
        check(ok, "Q=0 -> causal running mean of V");
        logline("INFO", "  t2=(%.3f,%.3f) expect(4.000,10.000)", out[4], out[5]);
    }

    // Case C: causal mask -> future V does not affect earlier outputs.
    {
        int T = 2, H = 1, KVH = 1, D = 2;
        std::vector<float> Q(T * D, 0.1f), K(T * D, 0.1f);
        std::vector<float> Va = {1.0f, 1.0f, 5.0f, 5.0f};
        std::vector<float> Vb = {1.0f, 1.0f, 99.0f, 99.0f};
        std::vector<float> oa(T * D), ob(T * D);
        attention_ref(Q.data(), K.data(), Va.data(), oa.data(), T, H, KVH, D, scale);
        attention_ref(Q.data(), K.data(), Vb.data(), ob.data(), T, H, KVH, D, scale);
        bool ok = approx(oa[0], ob[0], 1e-6f) && approx(oa[1], ob[1], 1e-6f);
        check(ok, "changing V[1] leaves out[0] unchanged (causal)");
    }

    if (g_fail == 0) { logline("INFO", "ALL CHECKS PASSED"); return 0; }
    logline("INFO", "%d CHECK(S) FAILED", g_fail);
    return 1;
}

// Full Llama decoder layer reference + self-test (CPU, fp32). Mirrors the GPU
// layer_forward assembly (PLAN section 7). Validates the wiring end to end.
// Weights are row-major [in, out]. Single-line comments only (linter rule).
#include <cstdio>
#include <cmath>
#include <cstdarg>
#include <vector>

struct Cfg { int dim, n_heads, n_kv_heads, head_dim, ffn_dim; float eps, rope_base; };

static void rmsnorm(const float* x, const float* w, float* out, int rows, int dim, float eps) {
    for (int r = 0; r < rows; ++r) {
        double ss = 0.0;
        for (int j = 0; j < dim; ++j) ss += (double)x[r*dim+j] * x[r*dim+j];
        float inv = 1.0f / sqrtf((float)(ss / dim) + eps);
        for (int j = 0; j < dim; ++j) out[r*dim+j] = x[r*dim+j] * inv * w[j];
    }
}

// C[m,n] = A[m,k] * B[k,n], all row-major.
static void matmul(const float* A, const float* B, float* C, int m, int n, int k) {
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (int p = 0; p < k; ++p) acc += A[i*k+p] * B[p*n+j];
            C[i*n+j] = acc;
        }
}

static void rope(float* x, const int* pos, int T, int heads, int hd, float base) {
    int half = hd / 2;
    for (int blk = 0; blk < T*heads; ++blk) {
        int t = blk / heads; float* v = x + (size_t)blk*hd;
        for (int j = 0; j < half; ++j) {
            float ang = pos[t] * powf(base, -2.0f*j/hd);
            float c = cosf(ang), s = sinf(ang);
            float a = v[2*j], b = v[2*j+1];
            v[2*j] = a*c - b*s; v[2*j+1] = a*s + b*c;
        }
    }
}

static void attention(const float* Q, const float* K, const float* V, float* out,
                      int T, int H, int KVH, int D, float scale) {
    int group = H / KVH; std::vector<float> sc(T);
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < H; ++h) {
            int kvh = h / group; const float* q = Q + ((size_t)t*H+h)*D;
            float m = -1e30f;
            for (int j = 0; j <= t; ++j) {
                const float* kk = K + ((size_t)j*KVH+kvh)*D;
                float s = 0; for (int d = 0; d < D; ++d) s += q[d]*kk[d];
                sc[j] = s*scale; m = fmaxf(m, sc[j]);
            }
            float den = 0; for (int j = 0; j <= t; ++j) { sc[j] = expf(sc[j]-m); den += sc[j]; }
            float* o = out + ((size_t)t*H+h)*D;
            for (int d = 0; d < D; ++d) o[d] = 0;
            for (int j = 0; j <= t; ++j) {
                const float* vv = V + ((size_t)j*KVH+kvh)*D; float p = sc[j]/den;
                for (int d = 0; d < D; ++d) o[d] += p*vv[d];
            }
        }
}

static float silu(float v) { return v / (1.0f + expf(-v)); }

struct W {
    const float *attn_norm, *wq, *wk, *wv, *wo, *ffn_norm, *wgate, *wup, *wdown;
};

// One decoder layer, in-place on hidden[T*dim].
static void layer_forward(const Cfg& c, const W& w, float* hidden, const int* pos, int T) {
    int qd = c.n_heads * c.head_dim, kvd = c.n_kv_heads * c.head_dim;
    float scale = 1.0f / sqrtf((float)c.head_dim);
    std::vector<float> xn(T*c.dim), q(T*qd), k(T*kvd), v(T*kvd), att(T*qd), proj(T*c.dim);
    std::vector<float> gate(T*c.ffn_dim), up(T*c.ffn_dim);

    rmsnorm(hidden, w.attn_norm, xn.data(), T, c.dim, c.eps);
    matmul(xn.data(), w.wq, q.data(), T, qd, c.dim);
    matmul(xn.data(), w.wk, k.data(), T, kvd, c.dim);
    matmul(xn.data(), w.wv, v.data(), T, kvd, c.dim);
    rope(q.data(), pos, T, c.n_heads, c.head_dim, c.rope_base);
    rope(k.data(), pos, T, c.n_kv_heads, c.head_dim, c.rope_base);
    attention(q.data(), k.data(), v.data(), att.data(), T, c.n_heads, c.n_kv_heads, c.head_dim, scale);
    matmul(att.data(), w.wo, proj.data(), T, c.dim, qd);
    for (int i = 0; i < T*c.dim; ++i) hidden[i] += proj[i];

    rmsnorm(hidden, w.ffn_norm, xn.data(), T, c.dim, c.eps);
    matmul(xn.data(), w.wgate, gate.data(), T, c.ffn_dim, c.dim);
    matmul(xn.data(), w.wup, up.data(), T, c.ffn_dim, c.dim);
    for (int i = 0; i < T*c.ffn_dim; ++i) gate[i] = silu(gate[i]) * up[i];
    matmul(gate.data(), w.wdown, proj.data(), T, c.dim, c.ffn_dim);
    for (int i = 0; i < T*c.dim; ++i) hidden[i] += proj[i];
}

static int g_fail = 0;
static void logline(const char* lv, const char* fmt, ...) {
    printf("[%-4s] ", lv); va_list ap; va_start(ap, fmt); vprintf(fmt, ap); va_end(ap); printf("\n");
}
static void check(bool c, const char* what) {
    if (c) logline("PASS", "%s", what); else { logline("FAIL", "%s", what); ++g_fail; }
}

int main() {
    logline("INFO", "Full decoder layer assembly test");
    Cfg c{32, 4, 2, 8, 64, 1e-6f, 10000.0f};
    int qd = c.n_heads*c.head_dim, kvd = c.n_kv_heads*c.head_dim;
    int T = 5;
    std::vector<int> pos = {0,1,2,3,4};

    auto fill = [](std::vector<float>& x, int seed) {
        for (size_t i = 0; i < x.size(); ++i) x[i] = 0.05f*(float)((int)((i*7+seed)%23) - 11);
    };
    std::vector<float> an(c.dim,1.0f), fn(c.dim,1.0f);
    std::vector<float> wq(c.dim*qd), wk(c.dim*kvd), wv(c.dim*kvd), wo(qd*c.dim);
    std::vector<float> wg(c.dim*c.ffn_dim), wu(c.dim*c.ffn_dim), wd(c.ffn_dim*c.dim);
    fill(wq,1); fill(wk,2); fill(wv,3); fill(wo,4); fill(wg,5); fill(wu,6); fill(wd,7);

    W w{an.data(), wq.data(), wk.data(), wv.data(), wo.data(), fn.data(),
        wg.data(), wu.data(), wd.data()};

    std::vector<float> h0(T*c.dim);
    fill(h0, 9);

    // Case 1: general run stays finite.
    {
        std::vector<float> h = h0;
        layer_forward(c, w, h.data(), pos.data(), T);
        bool ok = true;
        for (float v : h) if (!std::isfinite(v)) ok = false;
        check(ok, "layer output is finite");
        logline("INFO", "  h_out[0]=%.5f h_out[last]=%.5f", h[0], h[T*c.dim-1]);
    }

    // Case 2: zero output projections (wo=0, wdown=0) -> residual identity.
    {
        std::vector<float> zo(qd*c.dim, 0.0f), zd(c.ffn_dim*c.dim, 0.0f);
        W w2 = w; w2.wo = zo.data(); w2.wdown = zd.data();
        std::vector<float> h = h0;
        layer_forward(c, w2, h.data(), pos.data(), T);
        bool ok = true;
        for (int i = 0; i < T*c.dim; ++i) if (fabsf(h[i]-h0[i]) > 1e-5f) ok = false;
        check(ok, "wo=0, wdown=0 -> output == input (residual identity)");
    }

    if (g_fail == 0) { logline("INFO", "ALL CHECKS PASSED"); return 0; }
    logline("INFO", "%d CHECK(S) FAILED", g_fail);
    return 1;
}

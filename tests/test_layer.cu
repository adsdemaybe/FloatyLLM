// Unit test for the full decoder layer. RUN on a GPU. Compares GPU layer_forward
// vs an fp32 host reference (chained fp16 -> looser tolerance).
#include "layer.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

// --- fp32 host reference (mirrors src/layer.cu) ---
static void h_rmsnorm(const float* x, const float* w, float* o, int R, int d, float eps) {
    for (int r = 0; r < R; ++r) {
        double ss = 0; for (int j = 0; j < d; ++j) ss += (double)x[r*d+j]*x[r*d+j];
        float inv = 1.0f/sqrtf((float)(ss/d)+eps);
        for (int j = 0; j < d; ++j) o[r*d+j] = x[r*d+j]*inv*w[j];
    }
}
static void h_mm(const float* A, const float* B, float* C, int m, int n, int k) {
    for (int i = 0; i < m; ++i) for (int j = 0; j < n; ++j) {
        float a = 0; for (int p = 0; p < k; ++p) a += A[i*k+p]*B[p*n+j]; C[i*n+j] = a; }
}
static void h_rope(float* x, const int* pos, int T, int H, int hd, float base) {
    int half = hd/2;
    for (int b = 0; b < T*H; ++b) { int t = b/H; float* v = x+(size_t)b*hd;
        for (int j = 0; j < half; ++j) { float ang = pos[t]*powf(base,-2.0f*j/hd);
            float c = cosf(ang), s = sinf(ang); float a = v[2*j], bb = v[2*j+1];
            v[2*j] = a*c-bb*s; v[2*j+1] = a*s+bb*c; } }
}
static void h_attn(const float* Q, const float* K, const float* V, float* out,
                   int T, int H, int KVH, int D, float scale) {
    int g = H/KVH; std::vector<float> sc(T);
    for (int t = 0; t < T; ++t) for (int h = 0; h < H; ++h) { int kvh = h/g;
        const float* q = Q+((size_t)t*H+h)*D; float m = -1e30f;
        for (int j = 0; j <= t; ++j) { const float* k = K+((size_t)j*KVH+kvh)*D;
            float s = 0; for (int d = 0; d < D; ++d) s += q[d]*k[d]; sc[j] = s*scale; m = fmaxf(m,sc[j]); }
        float den = 0; for (int j = 0; j <= t; ++j) { sc[j] = expf(sc[j]-m); den += sc[j]; }
        float* o = out+((size_t)t*H+h)*D; for (int d = 0; d < D; ++d) o[d] = 0;
        for (int j = 0; j <= t; ++j) { const float* v = V+((size_t)j*KVH+kvh)*D; float p = sc[j]/den;
            for (int d = 0; d < D; ++d) o[d] += p*v[d]; } }
}
static float h_silu(float v) { return v/(1.0f+expf(-v)); }

int main() {
    LlamaConfig cfg{32, 4, 2, 8, 64, 1e-6f, 10000.0f};
    int T = 5, dim = cfg.dim, qd = cfg.n_heads*cfg.head_dim, kvd = cfg.n_kv_heads*cfg.head_dim, ffn = cfg.ffn_dim;
    std::vector<int> pos = {0,1,2,3,4};

    auto fill = [](std::vector<float>& x, int s) {
        for (size_t i = 0; i < x.size(); ++i) x[i] = 0.05f*(float)((int)((i*7+s)%23) - 11); };
    std::vector<float> an(dim,1.0f), fn(dim,1.0f);
    std::vector<float> wq(dim*qd), wk(dim*kvd), wv(dim*kvd), wo(qd*dim), wg(dim*ffn), wu(dim*ffn), wd(ffn*dim);
    fill(wq,1); fill(wk,2); fill(wv,3); fill(wo,4); fill(wg,5); fill(wu,6); fill(wd,7);
    std::vector<float> hf(T*dim); fill(hf,9);

    // Host reference (in-place on a copy).
    std::vector<float> hexp = hf;
    {
        std::vector<float> xn(T*dim), q(T*qd), k(T*kvd), v(T*kvd), att(T*qd), proj(T*dim), gate(T*ffn), up(T*ffn);
        float scale = 1.0f/sqrtf((float)cfg.head_dim);
        h_rmsnorm(hexp.data(), an.data(), xn.data(), T, dim, cfg.eps);
        h_mm(xn.data(), wq.data(), q.data(), T, qd, dim);
        h_mm(xn.data(), wk.data(), k.data(), T, kvd, dim);
        h_mm(xn.data(), wv.data(), v.data(), T, kvd, dim);
        h_rope(q.data(), pos.data(), T, cfg.n_heads, cfg.head_dim, cfg.rope_base);
        h_rope(k.data(), pos.data(), T, cfg.n_kv_heads, cfg.head_dim, cfg.rope_base);
        h_attn(q.data(), k.data(), v.data(), att.data(), T, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, scale);
        h_mm(att.data(), wo.data(), proj.data(), T, dim, qd);
        for (int i = 0; i < T*dim; ++i) hexp[i] += proj[i];
        h_rmsnorm(hexp.data(), fn.data(), xn.data(), T, dim, cfg.eps);
        h_mm(xn.data(), wg.data(), gate.data(), T, ffn, dim);
        h_mm(xn.data(), wu.data(), up.data(), T, ffn, dim);
        for (int i = 0; i < T*ffn; ++i) gate[i] = h_silu(gate[i])*up[i];
        h_mm(gate.data(), wd.data(), proj.data(), T, dim, ffn);
        for (int i = 0; i < T*dim; ++i) hexp[i] += proj[i];
    }

    // Device setup.
    auto to_h = [](const std::vector<float>& f) { std::vector<__half> h(f.size());
        for (size_t i = 0; i < f.size(); ++i) h[i] = __float2half(f[i]); return h; };
    auto up_dev = [](const std::vector<__half>& h) { __half* d; cudaMalloc(&d, h.size()*sizeof(__half));
        cudaMemcpy(d, h.data(), h.size()*sizeof(__half), cudaMemcpyHostToDevice); return d; };

    LayerWeights w{ up_dev(to_h(an)), up_dev(to_h(wq)), up_dev(to_h(wk)), up_dev(to_h(wv)),
                    up_dev(to_h(wo)), up_dev(to_h(fn)), up_dev(to_h(wg)), up_dev(to_h(wu)), up_dev(to_h(wd)) };
    __half* dh = up_dev(to_h(hf));
    int* dpos; cudaMalloc(&dpos, pos.size()*sizeof(int));
    cudaMemcpy(dpos, pos.data(), pos.size()*sizeof(int), cudaMemcpyHostToDevice);

    LayerScratch s;
    cudaMalloc(&s.xn, T*dim*sizeof(__half)); cudaMalloc(&s.q, T*qd*sizeof(__half));
    cudaMalloc(&s.k, T*kvd*sizeof(__half)); cudaMalloc(&s.v, T*kvd*sizeof(__half));
    cudaMalloc(&s.att, T*qd*sizeof(__half)); cudaMalloc(&s.proj, T*dim*sizeof(__half));
    cudaMalloc(&s.gate, T*ffn*sizeof(__half)); cudaMalloc(&s.up, T*ffn*sizeof(__half));

    Gemm g; gemm_create(&g);
    layer_forward(cfg, w, dh, dpos, T, s, &g, 0);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 2; }

    std::vector<__half> hout(T*dim);
    cudaMemcpy(hout.data(), dh, hout.size()*sizeof(__half), cudaMemcpyDeviceToHost);

    // fp16 intermediates chained through ~15 ops drift vs the fp32 reference, worst
    // on the last token (most attention context). Loose bound here; the authoritative
    // layer check is the future bit-for-bit-ish match against llama.cpp (also fp16).
    int fails = 0;
    float max_err = 0.0f;
    for (int i = 0; i < T*dim; ++i) {
        float got = __half2float(hout[i]);
        max_err = fmaxf(max_err, fabsf(got - hexp[i]));
        float tol = 1.5e-1f * (fabsf(hexp[i]) + 1.0f);
        if (fabsf(got - hexp[i]) > tol) { if (fails < 10) printf("mismatch i=%d got=%f exp=%f\n", i, got, hexp[i]); ++fails; }
    }
    printf("max abs err (fp16 vs fp32 ref) = %.4f\n", max_err);
    gemm_destroy(&g);
    if (fails) { printf("FAIL: %d/%d\n", fails, T*dim); return 1; }
    printf("PASS: full layer T=%d dim=%d (fp16 vs fp32 ref)\n", T, dim);
    return 0;
}

// Unit test for MoE MLP. RUN on a GPU. With identical experts, the top-k routed
// weighted sum renormalizes to 1, so MoE output == a single dense SwiGLU.
#include "moe.h"
#include "elementwise.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

static __half* up_half(const std::vector<float>& f) {
    std::vector<__half> h(f.size());
    for (size_t i = 0; i < f.size(); ++i) h[i] = __float2half(f[i]);
    __half* d; cudaMalloc(&d, h.size() * sizeof(__half));
    cudaMemcpy(d, h.data(), h.size() * sizeof(__half), cudaMemcpyHostToDevice);
    return d;
}

int main() {
    const int T = 4, dim = 32, ffn = 64, E = 4, topk = 2;

    auto fill = [](std::vector<float>& x, float s, int seed) {
        for (size_t i = 0; i < x.size(); ++i) x[i] = s * (float)((int)((i * 7 + seed) % 23) - 11);
    };
    std::vector<float> xf(T*dim), wgf(dim*ffn), wuf(dim*ffn), wdf(ffn*dim), wgatef(dim*E);
    fill(xf, 0.1f, 1); fill(wgf, 0.05f, 2); fill(wuf, 0.05f, 3); fill(wdf, 0.05f, 4); fill(wgatef, 0.1f, 5);

    __half* x = up_half(xf);
    __half* Wg = up_half(wgf); __half* Wu = up_half(wuf); __half* Wd = up_half(wdf);
    __half* Wgate = up_half(wgatef);

    // All experts identical (same device pointers).
    std::vector<const __half*> wgate(E, Wg), wup(E, Wu), wdown(E, Wd);

    __half *gate, *up, *ye, *out, *logits; float* route_w;
    cudaMalloc(&gate, T*ffn*sizeof(__half)); cudaMalloc(&up, T*ffn*sizeof(__half));
    cudaMalloc(&ye, T*dim*sizeof(__half)); cudaMalloc(&out, T*dim*sizeof(__half));
    cudaMalloc(&logits, T*E*sizeof(__half)); cudaMalloc(&route_w, T*E*sizeof(float));

    Gemm g; gemm_create(&g);
    moe_router(&g, x, Wgate, logits, route_w, T, dim, E, topk, 0);
    std::vector<int> active(E); for (int e = 0; e < E; ++e) active[e] = e;
    moe_mlp(&g, x, wgate.data(), wup.data(), wdown.data(), route_w, out, gate, up, ye,
            T, dim, ffn, E, active.data(), E, 0);

    // Reference: single dense SwiGLU with the shared weights.
    __half *rg, *ru, *rh, *ry;
    cudaMalloc(&rg, T*ffn*sizeof(__half)); cudaMalloc(&ru, T*ffn*sizeof(__half));
    cudaMalloc(&rh, T*ffn*sizeof(__half)); cudaMalloc(&ry, T*dim*sizeof(__half));
    gemm_rowmajor(&g, x, Wg, rg, T, ffn, dim, 0);
    gemm_rowmajor(&g, x, Wu, ru, T, ffn, dim, 0);
    silu(rg, rh, T*ffn, 0);
    elementwise_mul(rh, ru, rh, T*ffn, 0);
    gemm_rowmajor(&g, rh, Wd, ry, T, dim, ffn, 0);
    cudaDeviceSynchronize();

    std::vector<__half> ho(T*dim), hr(T*dim);
    cudaMemcpy(ho.data(), out, T*dim*sizeof(__half), cudaMemcpyDeviceToHost);
    cudaMemcpy(hr.data(), ry, T*dim*sizeof(__half), cudaMemcpyDeviceToHost);

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 2; }

    int fails = 0; float maxd = 0;
    for (int i = 0; i < T*dim; ++i) {
        float a = __half2float(ho[i]), b = __half2float(hr[i]);
        maxd = fmaxf(maxd, fabsf(a - b));
        if (fabsf(a - b) > 2e-2f * (fabsf(b) + 1.0f)) { if (fails < 8) printf("mismatch i=%d moe=%f ref=%f\n", i, a, b); ++fails; }
    }
    printf("MoE vs dense-SwiGLU (identical experts) max_diff=%.5f\n", maxd);
    gemm_destroy(&g);
    if (fails) { printf("FAIL: %d/%d\n", fails, T*dim); return 1; }
    printf("PASS: MoE MLP (E=%d, top-%d) == dense SwiGLU with identical experts\n", E, topk);
    return 0;
}

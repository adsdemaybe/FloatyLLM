// Unit test for the dense runner. RUN on a GPU. Synthetic tiny model: checks
// forward_logits produces finite logits, a valid argmax, and is deterministic.
#include "runner.h"
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
    LlamaConfig cfg{32, 4, 2, 8, 64, 1e-5f, 10000.0f};
    const int n_layers = 4, T = 5, vocab = 50;
    LayerBlob blob = layer_blob_layout(cfg);
    const size_t per = blob.total_elems;

    auto fill = [](std::vector<float>& x, float s, int seed) {
        for (size_t i = 0; i < x.size(); ++i) x[i] = s * (float)((int)((i * 7 + seed) % 23) - 11);
    };

    std::vector<float> embd(vocab * cfg.dim), outw(vocab * cfg.dim), fnorm(cfg.dim, 1.0f);
    fill(embd, 0.05f, 1); fill(outw, 0.05f, 2);
    RunnerWeights rw{ up_half(embd), up_half(fnorm), up_half(outw) };

    // Pinned layer weights.
    __half* hw; cudaHostAlloc((void**)&hw, per * n_layers * sizeof(__half), cudaHostAllocDefault);
    for (size_t i = 0; i < per * n_layers; ++i) hw[i] = __float2half(0.02f * (float)((int)(i % 37) - 18));

    std::vector<int> ids = {1, 7, 3, 42, 9};
    int* d_ids; cudaMalloc(&d_ids, T * sizeof(int));
    cudaMemcpy(d_ids, ids.data(), T * sizeof(int), cudaMemcpyHostToDevice);

    RunnerBufs bufs;
    cudaMalloc(&bufs.hidden, T * cfg.dim * sizeof(__half));
    cudaMalloc(&bufs.normed, T * cfg.dim * sizeof(__half));
    cudaMalloc(&bufs.positions, T * sizeof(int));
    cudaMalloc(&bufs.logits, vocab * sizeof(__half));

    int qd = cfg.n_heads*cfg.head_dim, kvd = cfg.n_kv_heads*cfg.head_dim, ffn = cfg.ffn_dim;
    LayerScratch s;
    cudaMalloc(&s.xn, T*cfg.dim*sizeof(__half)); cudaMalloc(&s.q, T*qd*sizeof(__half));
    cudaMalloc(&s.k, T*kvd*sizeof(__half)); cudaMalloc(&s.v, T*kvd*sizeof(__half));
    cudaMalloc(&s.att, T*qd*sizeof(__half)); cudaMalloc(&s.proj, T*cfg.dim*sizeof(__half));
    cudaMalloc(&s.gate, T*ffn*sizeof(__half)); cudaMalloc(&s.up, T*ffn*sizeof(__half));

    SlotPool pool; slotpool_create(&pool, 3, per);
    cudaStream_t cs, ms; cudaStreamCreate(&cs); cudaStreamCreate(&ms);
    Gemm g; gemm_create(&g);

    auto run = [&](std::vector<__half>& out) {
        forward_logits(cfg, blob, hw, n_layers, 1, rw, d_ids, T, vocab, bufs, s, pool, &g, cs, ms);
        out.resize(vocab);
        cudaMemcpy(out.data(), bufs.logits, vocab * sizeof(__half), cudaMemcpyDeviceToHost);
    };

    std::vector<__half> l1, l2;
    run(l1); run(l2);

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 2; }

    int fails = 0;
    bool finite = true; int argmax = 0; float best = -1e30f; bool determ = true;
    for (int v = 0; v < vocab; ++v) {
        float x = __half2float(l1[v]);
        if (!std::isfinite(x)) finite = false;
        if (x > best) { best = x; argmax = v; }
        if (fabsf(__half2float(l1[v]) - __half2float(l2[v])) > 1e-3f) determ = false;
    }
    if (!finite) { printf("FAIL: non-finite logits\n"); ++fails; }
    if (argmax < 0 || argmax >= vocab) { printf("FAIL: argmax out of range\n"); ++fails; }
    if (!determ) { printf("FAIL: non-deterministic\n"); ++fails; }
    printf("logits finite=%d argmax=%d deterministic=%d\n", (int)finite, argmax, (int)determ);

    slotpool_destroy(&pool); gemm_destroy(&g);
    if (fails) return 1;
    printf("PASS: runner forward_logits (%d layers, vocab=%d)\n", n_layers, vocab);
    return 0;
}

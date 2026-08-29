// Unit test for the streaming scheduler. RUN on a GPU. Streams N layers through
// the ring/pipeline and checks the result equals the resident (no-streaming) path.
#include "stream.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

int main() {
    LlamaConfig cfg{32, 4, 2, 8, 64, 1e-5f, 10000.0f};
    const int n_layers = 6, T = 4;
    LayerBlob blob = layer_blob_layout(cfg);
    const size_t per = blob.total_elems;
    const size_t total = per * n_layers;

    // Pinned host weights (required for async DMA overlap).
    __half* hw = nullptr;
    cudaHostAlloc((void**)&hw, total * sizeof(__half), cudaHostAllocDefault);
    for (size_t i = 0; i < total; ++i) hw[i] = __float2half(0.02f * (float)((int)(i % 37) - 18));

    std::vector<int> pos = {0, 1, 2, 3};
    int* dpos; cudaMalloc(&dpos, pos.size() * sizeof(int));
    cudaMemcpy(dpos, pos.data(), pos.size() * sizeof(int), cudaMemcpyHostToDevice);

    std::vector<__half> h_init(T * cfg.dim);
    for (int i = 0; i < T * cfg.dim; ++i) h_init[i] = __float2half(0.05f * (float)((i % 19) - 9));

    // Shared scratch + gemm.
    int qd = cfg.n_heads * cfg.head_dim, kvd = cfg.n_kv_heads * cfg.head_dim, ffn = cfg.ffn_dim;
    LayerScratch s;
    cudaMalloc(&s.xn, T*cfg.dim*sizeof(__half)); cudaMalloc(&s.q, T*qd*sizeof(__half));
    cudaMalloc(&s.k, T*kvd*sizeof(__half)); cudaMalloc(&s.v, T*kvd*sizeof(__half));
    cudaMalloc(&s.att, T*qd*sizeof(__half)); cudaMalloc(&s.proj, T*cfg.dim*sizeof(__half));
    cudaMalloc(&s.gate, T*ffn*sizeof(__half)); cudaMalloc(&s.up, T*ffn*sizeof(__half));
    Gemm g; gemm_create(&g);

    // --- resident path: all weights in VRAM, no streaming ---
    __half* d_res_w; cudaMalloc(&d_res_w, total * sizeof(__half));
    cudaMemcpy(d_res_w, hw, total * sizeof(__half), cudaMemcpyHostToDevice);
    __half* d_hidden_res; cudaMalloc(&d_hidden_res, h_init.size()*sizeof(__half));
    cudaMemcpy(d_hidden_res, h_init.data(), h_init.size()*sizeof(__half), cudaMemcpyHostToDevice);
    for (int L = 0; L < n_layers; ++L) {
        LayerWeights w = weights_from_blob(d_res_w + (size_t)L * per, blob);
        layer_forward(cfg, w, d_hidden_res, dpos, T, s, &g, 0);
    }
    cudaDeviceSynchronize();
    std::vector<__half> out_res(T*cfg.dim);
    cudaMemcpy(out_res.data(), d_hidden_res, out_res.size()*sizeof(__half), cudaMemcpyDeviceToHost);

    // --- streamed path: ring of 3 slots, copy/compute overlap ---
    SlotPool pool; slotpool_create(&pool, 3, per);
    cudaStream_t copy_stream, compute_stream;
    cudaStreamCreate(&copy_stream); cudaStreamCreate(&compute_stream);
    __half* d_hidden_str; cudaMalloc(&d_hidden_str, h_init.size()*sizeof(__half));
    cudaMemcpy(d_hidden_str, h_init.data(), h_init.size()*sizeof(__half), cudaMemcpyHostToDevice);
    stream_forward(cfg, blob, hw, n_layers, 1, d_hidden_str, dpos, T, s, pool, &g, copy_stream, compute_stream);
    std::vector<__half> out_str(T*cfg.dim);
    cudaMemcpy(out_str.data(), d_hidden_str, out_str.size()*sizeof(__half), cudaMemcpyDeviceToHost);

    // --- batched streaming: 3 layers per DMA (one big transfer), ring of 2 batches ---
    SlotPool poolb; slotpool_create(&poolb, 2, (size_t)3 * per);
    __half* d_hidden_b; cudaMalloc(&d_hidden_b, h_init.size()*sizeof(__half));
    cudaMemcpy(d_hidden_b, h_init.data(), h_init.size()*sizeof(__half), cudaMemcpyHostToDevice);
    stream_forward(cfg, blob, hw, n_layers, 3, d_hidden_b, dpos, T, s, poolb, &g, copy_stream, compute_stream);
    std::vector<__half> out_batch(T*cfg.dim);
    cudaMemcpy(out_batch.data(), d_hidden_b, out_batch.size()*sizeof(__half), cudaMemcpyDeviceToHost);
    slotpool_destroy(&poolb);

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 2; }

    float batch_diff = 0.0f;
    for (int i = 0; i < T*cfg.dim; ++i)
        batch_diff = fmaxf(batch_diff, fabsf(__half2float(out_res[i]) - __half2float(out_batch[i])));
    printf("batched(3/DMA) vs resident: max_diff=%.6f\n", batch_diff);
    if (batch_diff > 1e-3f) { printf("FAIL: batched streaming differs\n"); return 1; }

    int fails = 0; float max_diff = 0.0f; bool finite = true;
    for (int i = 0; i < T*cfg.dim; ++i) {
        float r = __half2float(out_res[i]), st = __half2float(out_str[i]);
        if (!std::isfinite(r) || !std::isfinite(st)) finite = false;
        float d = fabsf(r - st); max_diff = fmaxf(max_diff, d);
        if (d > 1e-3f) { if (fails < 8) printf("mismatch i=%d res=%f str=%f\n", i, r, st); ++fails; }
    }
    printf("streamed vs resident: max_diff=%.6f finite=%d\n", max_diff, (int)finite);

    slotpool_destroy(&pool); gemm_destroy(&g);
    if (!finite) { printf("FAIL: non-finite output\n"); return 1; }
    if (fails) { printf("FAIL: %d/%d differ\n", fails, T*cfg.dim); return 1; }
    printf("PASS: streaming %d layers matches resident (ring=3)\n", n_layers);
    return 0;
}

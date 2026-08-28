// SemiLLM CLI: load a GGUF dense model, run the streamed forward for the given
// token ids, print the next-token argmax + top logits + GPU memory used.
// usage: semillm <model.gguf> <tok0> [tok1 ...]
#include "weights.h"
#include "runner.h"
#include "sampling.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <algorithm>

static void report_mem(const char* tag) {
    size_t freeb = 0, totalb = 0;
    cudaMemGetInfo(&freeb, &totalb);
    printf("[mem] %-14s free=%.2f GB / total=%.2f GB\n", tag,
           freeb / 1e9, totalb / 1e9);
}

int main(int argc, char** argv) {
    if (argc < 3) { printf("usage: %s <model.gguf> <tok0> [tok1 ...]\n", argv[0]); return 2; }

    std::vector<int> ids;
    for (int i = 2; i < argc; ++i) ids.push_back(atoi(argv[i]));
    int T = (int)ids.size();

    report_mem("start");
    printf("loading %s ...\n", argv[1]);
    GgufFile g; std::string err;
    if (!gguf_load(argv[1], &g, &err)) { printf("gguf_load: %s\n", err.c_str()); return 1; }
    printf("parsed: %zu tensors, %zu meta keys\n", g.tensors.size(), g.meta.size());

    LoadedModel m;
    if (!load_model(g, &m, &err)) { printf("load_model: %s\n", err.c_str()); return 1; }
    const LlamaConfig& cfg = m.cfg;
    printf("model: arch=%s layers=%d dim=%d heads=%d/%d head_dim=%d ffn=%d vocab=%d\n",
           m.arch.c_str(), m.n_layers, cfg.dim, cfg.n_heads, cfg.n_kv_heads,
           cfg.head_dim, cfg.ffn_dim, m.vocab);
    printf("per-layer blob = %.1f MB, whole model (fp16) = %.1f GB\n",
           m.blob.total_elems * 2 / 1e6, m.blob.total_elems * 2.0 * m.n_layers / 1e9);

    // Device buffers.
    int qd = cfg.n_heads*cfg.head_dim, kvd = cfg.n_kv_heads*cfg.head_dim, ffn = cfg.ffn_dim;
    RunnerBufs bufs;
    cudaMalloc(&bufs.hidden, (size_t)T*cfg.dim*sizeof(__half));
    cudaMalloc(&bufs.normed, (size_t)T*cfg.dim*sizeof(__half));
    cudaMalloc(&bufs.positions, T*sizeof(int));
    cudaMalloc(&bufs.logits, (size_t)m.vocab*sizeof(__half));
    LayerScratch s;
    cudaMalloc(&s.xn, (size_t)T*cfg.dim*sizeof(__half)); cudaMalloc(&s.q, (size_t)T*qd*sizeof(__half));
    cudaMalloc(&s.k, (size_t)T*kvd*sizeof(__half)); cudaMalloc(&s.v, (size_t)T*kvd*sizeof(__half));
    cudaMalloc(&s.att, (size_t)T*qd*sizeof(__half)); cudaMalloc(&s.proj, (size_t)T*cfg.dim*sizeof(__half));
    cudaMalloc(&s.gate, (size_t)T*ffn*sizeof(__half)); cudaMalloc(&s.up, (size_t)T*ffn*sizeof(__half));

    int n_slots = 4;
    SlotPool pool; slotpool_create(&pool, n_slots, m.blob.total_elems);
    printf("slot pool: %d slots x %.1f MB = %.2f GB VRAM (bounded working set)\n",
           n_slots, m.blob.total_elems*2/1e6, (double)n_slots*m.blob.total_elems*2/1e9);
    report_mem("after alloc");

    int* d_ids; cudaMalloc(&d_ids, T*sizeof(int));
    cudaMemcpy(d_ids, ids.data(), T*sizeof(int), cudaMemcpyHostToDevice);
    cudaStream_t cs, ms; cudaStreamCreate(&cs); cudaStreamCreate(&ms);
    Gemm gemm; gemm_create(&gemm);

    forward_logits(cfg, m.blob, m.h_layer_weights, m.n_layers, m.rw, d_ids, T, m.vocab,
                   bufs, s, pool, &gemm, cs, ms);

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 1; }

    std::vector<__half> logits(m.vocab);
    cudaMemcpy(logits.data(), bufs.logits, m.vocab*sizeof(__half), cudaMemcpyDeviceToHost);
    std::vector<float> lf(m.vocab);
    for (int v = 0; v < m.vocab; ++v) lf[v] = __half2float(logits[v]);

    int argmax = sample_greedy(lf.data(), m.vocab);
    printf("\ninput ids (%d):", T);
    for (int id : ids) printf(" %d", id);
    printf("\nnext-token argmax = %d  (logit %.4f)\n", argmax, lf[argmax]);

    std::vector<int> idx(m.vocab);
    for (int v = 0; v < m.vocab; ++v) idx[v] = v;
    std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                      [&](int a, int b){ return lf[a] > lf[b]; });
    printf("top-5:");
    for (int i = 0; i < 5; ++i) printf(" [%d]=%.3f", idx[i], lf[idx[i]]);
    printf("\n");
    report_mem("after forward");

    slotpool_destroy(&pool); gemm_destroy(&gemm); free_model(&m);
    return 0;
}

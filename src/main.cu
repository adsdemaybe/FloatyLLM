// SemiLLM CLI: load a GGUF dense model and greedily generate tokens via the
// streamed forward. usage: semillm <model.gguf> <n_generate> <tok0> [tok1 ...]
// Prints the generated token ids (detokenize with llama.cpp) + tokens/s + VRAM.
#include "weights.h"
#include "runner.h"
#include "sampling.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <chrono>

static void report_mem(const char* tag) {
    size_t freeb = 0, totalb = 0;
    cudaMemGetInfo(&freeb, &totalb);
    printf("[mem] %-13s free=%.2f GB / total=%.2f GB\n", tag, freeb / 1e9, totalb / 1e9);
}

int main(int argc, char** argv) {
    if (argc < 4) { printf("usage: %s <model.gguf> <n_generate> <tok0> [tok1 ...]\n", argv[0]); return 2; }
    int n_gen = atoi(argv[2]);
    std::vector<int> ids;
    for (int i = 3; i < argc; ++i) ids.push_back(atoi(argv[i]));
    int prompt_len = (int)ids.size();
    int max_T = prompt_len + n_gen;

    report_mem("start");
    GgufFile g; std::string err;
    if (!gguf_load(argv[1], &g, &err)) { printf("gguf_load: %s\n", err.c_str()); return 1; }
    LoadedModel m;
    if (!load_model(g, &m, &err)) { printf("load_model: %s\n", err.c_str()); return 1; }
    const LlamaConfig& cfg = m.cfg;
    printf("model: arch=%s layers=%d dim=%d heads=%d/%d ffn=%d vocab=%d | %.1f GB fp16\n",
           m.arch.c_str(), m.n_layers, cfg.dim, cfg.n_heads, cfg.n_kv_heads, cfg.ffn_dim,
           m.vocab, m.blob.total_elems * 2.0 * m.n_layers / 1e9);

    int qd = cfg.n_heads*cfg.head_dim, kvd = cfg.n_kv_heads*cfg.head_dim, ffn = cfg.ffn_dim;
    RunnerBufs bufs;
    cudaMalloc(&bufs.hidden, (size_t)max_T*cfg.dim*sizeof(__half));
    cudaMalloc(&bufs.normed, (size_t)max_T*cfg.dim*sizeof(__half));
    cudaMalloc(&bufs.positions, max_T*sizeof(int));
    cudaMalloc(&bufs.logits, (size_t)m.vocab*sizeof(__half));
    LayerScratch s;
    cudaMalloc(&s.xn, (size_t)max_T*cfg.dim*sizeof(__half)); cudaMalloc(&s.q, (size_t)max_T*qd*sizeof(__half));
    cudaMalloc(&s.k, (size_t)max_T*kvd*sizeof(__half)); cudaMalloc(&s.v, (size_t)max_T*kvd*sizeof(__half));
    cudaMalloc(&s.att, (size_t)max_T*qd*sizeof(__half)); cudaMalloc(&s.proj, (size_t)max_T*cfg.dim*sizeof(__half));
    cudaMalloc(&s.gate, (size_t)max_T*ffn*sizeof(__half)); cudaMalloc(&s.up, (size_t)max_T*ffn*sizeof(__half));

    int n_slots = 4;
    const char* slots_env = getenv("SEMILLM_SLOTS");
    if (slots_env) { int v = atoi(slots_env); if (v >= 2) n_slots = v; }
    SlotPool pool; slotpool_create(&pool, n_slots, m.blob.total_elems);
    printf("streamed weight working set = %d slots x %.1f MB = %.2f GB VRAM\n",
           n_slots, m.blob.total_elems*2/1e6, (double)n_slots*m.blob.total_elems*2/1e9);
    report_mem("after alloc");

    int* d_ids; cudaMalloc(&d_ids, max_T*sizeof(int));
    cudaStream_t cs, ms; cudaStreamCreate(&cs); cudaStreamCreate(&ms);
    Gemm gemm; gemm_create(&gemm);
    std::vector<__half> logits(m.vocab);
    std::vector<float> lf(m.vocab);

    // Greedy generation: re-run the growing prefix each step (no KV cache yet).
    printf("\ngenerating %d tokens (greedy)...\n", n_gen);
    auto t0 = std::chrono::steady_clock::now();
    for (int step = 0; step < n_gen; ++step) {
        int T = (int)ids.size();
        cudaMemcpy(d_ids, ids.data(), T*sizeof(int), cudaMemcpyHostToDevice);
        forward_logits(cfg, m.blob, m.h_layer_weights, m.n_layers, m.rw, d_ids, T, m.vocab,
                       bufs, s, pool, &gemm, cs, ms);
        cudaMemcpy(logits.data(), bufs.logits, m.vocab*sizeof(__half), cudaMemcpyDeviceToHost);
        for (int v = 0; v < m.vocab; ++v) lf[v] = __half2float(logits[v]);
        int next = sample_greedy(lf.data(), m.vocab);
        ids.push_back(next);
        printf("  step %d -> token %d\n", step, next);
    }
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 1; }

    printf("\ngenerated ids:");
    for (int i = prompt_len; i < (int)ids.size(); ++i) printf(" %d", ids[i]);
    printf("\nfull sequence:");
    for (int id : ids) printf(" %d", id);
    printf("\n%.2f tokens/s (%d tokens in %.2fs, no KV cache)\n", n_gen / secs, n_gen, secs);
    report_mem("after gen");

    slotpool_destroy(&pool); gemm_destroy(&gemm); free_model(&m);
    return 0;
}

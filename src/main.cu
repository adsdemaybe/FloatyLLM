// FloatyLLM CLI: load a GGUF dense model and greedily generate tokens via the
// streamed forward. usage: semillm <model.gguf> <n_generate> <tok0> [tok1 ...]
// Prints the generated token ids (detokenize with llama.cpp) + tokens/s + VRAM.
#include "weights.h"
#include "moe_model.h"
#include "runner.h"
#include "sampling.h"
#include "tokenizer.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <chrono>
#include <iostream>
#include <utility>
#include <functional>

// `semillm tokenize <model.gguf> <text...>` — encode text to ids using the native SPM
// tokenizer (verify vs `llama-tokenize --ids`). Also detokenizes back as a round-trip check.
static int cmd_tokenize(int argc, char** argv) {
    if (argc < 4) { printf("usage: %s tokenize <model.gguf> <text...>\n", argv[0]); return 2; }
    Tokenizer tok; std::string err;
    if (!tokenizer_load(argv[2], &tok, &err)) { printf("tokenizer_load: %s\n", err.c_str()); return 1; }
    std::string text = argv[3];
    for (int i = 4; i < argc; ++i) { text += " "; text += argv[i]; }
    std::vector<int> ids = tokenizer_encode(tok, text);
    printf("ids (%zu):", ids.size());
    for (int id : ids) printf(" %d", id);
    printf("\ndetok: '");
    for (int id : ids) fputs(tokenizer_piece(tok, id).c_str(), stdout);
    printf("'\n");
    tokenizer_free(&tok);
    return 0;
}

static void report_mem(const char* tag) {
    size_t freeb = 0, totalb = 0;
    cudaMemGetInfo(&freeb, &totalb);
    printf("[mem] %-13s free=%.2f GB / total=%.2f GB\n", tag, freeb / 1e9, totalb / 1e9);
}

// Shared chat loop for the TUI. Renders each user turn with the model's chat template,
// feeds only the new delta (KV persists), and streams the detokenized reply until EOG or
// n_pred. eval(ids,n) runs the forward + updates logits; sample() returns the next id;
// reset() clears the model KV. add_special (BOS) is added only on the first eval.
static void chat_repl(Tokenizer& tok, bool interactive, const std::string& prompt, int n_pred,
                      const std::function<bool(const int*, int)>& eval,
                      const std::function<int()>& sample,
                      const std::function<void()>& reset) {
    std::vector<std::pair<std::string, std::string>> msgs;
    std::string prev_fmt; int evald = 0; srand(1234);
    auto run_turn = [&](const std::string& user) {
        msgs.push_back({"user", user});
        std::string fmt = tokenizer_apply_chat(tok, msgs, true);
        if (fmt.empty()) fmt = prev_fmt + user;              // no chat template: raw text
        std::string delta = fmt.size() >= prev_fmt.size() ? fmt.substr(prev_fmt.size()) : fmt;
        std::vector<int> ids = tokenizer_encode(tok, delta, evald == 0);
        if (ids.empty()) { printf("[empty prompt]\n"); return; }
        if (!eval(ids.data(), (int)ids.size())) { printf("[eval error]\n"); return; }
        evald += (int)ids.size();
        std::string reply; auto t0 = std::chrono::steady_clock::now(); int steps = 0;
        for (; steps < n_pred; ++steps) {
            int next = sample();
            if (tokenizer_is_eog(tok, next)) break;
            std::string piece = tokenizer_piece(tok, next);
            fputs(piece.c_str(), stdout); fflush(stdout); reply += piece;
            if (!eval(&next, 1)) { printf("\n[eval error]\n"); break; }
            evald += 1;
        }
        double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        printf("\n\033[2m[%d tok, %.2f tok/s]\033[0m\n", steps, steps / (secs > 0 ? secs : 1));
        msgs.push_back({"assistant", reply});
        prev_fmt = tokenizer_apply_chat(tok, msgs, false);
    };
    if (interactive) {
        std::string line;
        while (true) {
            fputs("\n>>> ", stdout); fflush(stdout);
            if (!std::getline(std::cin, line)) break;
            if (line == "/bye" || line == "/exit") break;
            if (line == "/reset") { msgs.clear(); prev_fmt.clear(); evald = 0; reset(); printf("(context cleared)\n"); continue; }
            if (line.empty()) continue;
            run_turn(line);
        }
    } else {
        run_turn(prompt);
    }
}

// `floatyllm run <model.gguf> [-p TEXT | -it] [-n N] [--temp T] [--budget GB] [--ctx N]`
// vLLM-style flags + an ollama-style interactive chat TUI (multi-turn, KV persists).
// Works for BOTH MoE (hot-expert cache) and dense (streamed via the SlotPool ring).
static int cmd_run(int argc, char** argv) {
    std::string model, prompt; int n_pred = 256, ctx = 4096; float temp = 0.0f; double budget = 8.0;
    bool interactive = false, stream_flag = false;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-it" || a == "--interactive") interactive = true;
        else if (a == "--stream") stream_flag = true;   // stream a dense model from mmap (no host requant) -> huge models
        else if ((a == "-p" || a == "--prompt") && i+1 < argc) prompt = argv[++i];
        else if ((a == "-n" || a == "--n-predict") && i+1 < argc) n_pred = atoi(argv[++i]);
        else if (a == "--temp" && i+1 < argc) temp = (float)atof(argv[++i]);
        else if (a == "--budget" && i+1 < argc) budget = atof(argv[++i]);
        else if (a == "--ctx" && i+1 < argc) ctx = atoi(argv[++i]);
        else if (!a.empty() && a[0] != '-' && model.empty()) model = a;
        else { printf("unknown arg: %s\n", a.c_str()); return 2; }
    }
    if (model.empty()) { printf("usage: %s run <model.gguf> [-p TEXT | -it] [-n N] [--temp T] [--budget GB] [--ctx N]\n", argv[0]); return 2; }
    if (prompt.empty() && !interactive) interactive = true;   // no prompt => chat

    GgufFile g; std::string err;
    if (!gguf_load(model.c_str(), &g, &err)) { printf("gguf_load: %s\n", err.c_str()); return 1; }
    Tokenizer tok;
    if (!tokenizer_load(model.c_str(), &tok, &err)) { printf("tokenizer: %s\n", err.c_str()); gguf_close(&g); return 1; }

    // MoE always streams; dense streams from mmap only with --stream (else the host-requant
    // path below). Streaming dense enables models far bigger than host RAM (e.g. 405B).
    if (is_moe_model(g) || stream_flag) {
        LoadedMoeModel m;
        if (!load_moe_model(g, &m, &err)) { printf("load_moe_model: %s\n", err.c_str()); return 1; }
        MoeSession S;
        if (!moe_session_init(m, ctx, budget, &S, &err)) { printf("session: %s\n", err.c_str()); return 1; }
        bool dense = m.mcfg.n_experts == 1;
        if (dense) printf("\n  FloatyLLM · %s · dense (streamed from mmap) · %d layers · %.0f GB fp16 equiv\n",
                          m.arch.c_str(), m.n_layers, m.blob.total_elems * 2.0 / 1e9 * m.n_layers);
        else printf("\n  FloatyLLM · %s · MoE %d/%d experts · %d layers · %.0f GB fp16 streamed from disk\n",
                    m.arch.c_str(), m.mcfg.n_used, m.mcfg.n_experts, m.n_layers, m.blob.total_elems * 2.0 / 1e9 * m.n_layers);
        printf("  ctx=%d  budget=%.0f GB  temp=%.2f%s\n", ctx, budget, temp,
               interactive ? "   (/bye to exit, /reset to clear context)" : "");
        auto eval = [&](const int* ids, int n) { return moe_session_eval(S, ids, n, &err); };
        auto sample = [&]() { return moe_session_sample(S, temp, (float)rand() / ((float)RAND_MAX + 1.0f)); };
        auto reset = [&]() { moe_session_reset(S); };
        chat_repl(tok, interactive, prompt, n_pred, eval, sample, reset);
        long uses = S.cache.hits + S.cache.misses;
        if (uses > 0) printf("\n[hot-expert cache: %.1f%% hit (%ld/%ld), %ld experts streamed]\n",
                             100.0 * S.cache.hits / uses, S.cache.hits, uses, S.cache.misses);
        moe_session_free(&S); free_moe_model(&m);
        tokenizer_free(&tok); gguf_close(&g);
        return 0;
    }

    // Dense: stream Q8_0 layer blobs through the SlotPool ring (forward_logits_cached).
    int q_bits = 8; const char* qb = getenv("SEMILLM_QBITS"); if (qb && atoi(qb) == 4) q_bits = 4;
    LoadedModel m;
    if (!load_model(g, &m, q_bits, &err)) { printf("load_model: %s\n", err.c_str()); gguf_close(&g); return 1; }
    const LlamaConfig& cfg = m.cfg;
    int max_T = ctx, qd = cfg.n_heads*cfg.head_dim, kvd = cfg.n_kv_heads*cfg.head_dim, ffn = cfg.ffn_dim;
    RunnerBufs bufs;
    cudaMalloc(&bufs.hidden, (size_t)max_T*cfg.dim*2); cudaMalloc(&bufs.normed, (size_t)max_T*cfg.dim*2);
    cudaMalloc(&bufs.positions, max_T*4); cudaMalloc(&bufs.logits, (size_t)m.vocab*2);
    cudaMalloc(&bufs.arena, (size_t)m.blob.total_elems*2);
    LayerScratch s;
    cudaMalloc(&s.xn,(size_t)max_T*cfg.dim*2); cudaMalloc(&s.q,(size_t)max_T*qd*2); cudaMalloc(&s.k,(size_t)max_T*kvd*2);
    cudaMalloc(&s.v,(size_t)max_T*kvd*2); cudaMalloc(&s.att,(size_t)max_T*qd*2); cudaMalloc(&s.proj,(size_t)max_T*cfg.dim*2);
    cudaMalloc(&s.gate,(size_t)max_T*ffn*2); cudaMalloc(&s.up,(size_t)max_T*ffn*2);
    int n_slots = 4, batch_layers = 1;
    const char* se = getenv("SEMILLM_SLOTS"); if (se && atoi(se) >= 2) n_slots = atoi(se);
    const char* be = getenv("SEMILLM_BATCH"); if (be && atoi(be) >= 1) batch_layers = atoi(be);
    size_t slot_elems = ((size_t)batch_layers * m.q8_layer_bytes + 1) / 2;
    SlotPool pool; slotpool_create(&pool, n_slots, slot_elems);
    KVCache kv; kv.n_layers = m.n_layers; kv.max_T = max_T; kv.kvd = kvd; kv.len = 0;
    cudaMalloc(&kv.K, (size_t)m.n_layers*max_T*kvd*2); cudaMalloc(&kv.V, (size_t)m.n_layers*max_T*kvd*2);
    int* d_ids; cudaMalloc(&d_ids, max_T*4);
    cudaStream_t cs, cms; cudaStreamCreate(&cs); cudaStreamCreate(&cms);
    Gemm gemm; gemm_create(&gemm);
    std::vector<__half> hl(m.vocab); std::vector<float> lf(m.vocab);

    printf("\n  FloatyLLM · %s · dense · %d layers · %.1f GB Q%d streamed (%d-slot ring)\n",
           m.arch.c_str(), m.n_layers, m.q8_layer_bytes / 1e9 * m.n_layers, m.q_bits, n_slots);
    printf("  ctx=%d  temp=%.2f%s\n", ctx, temp,
           interactive ? "   (/bye to exit, /reset to clear context)" : "");

    auto eval = [&](const int* ids, int n) -> bool {
        if (kv.len + n > max_T) { printf("[ctx exceeded]\n"); return false; }
        cudaMemcpy(d_ids, ids, n*4, cudaMemcpyHostToDevice);
        forward_logits_cached(cfg, m.blob, m.h_layer_q8, m.q8_layer_bytes, m.q_bits, m.n_layers, batch_layers,
                              m.rw, d_ids, n, kv.len, kv, m.vocab, bufs, s, pool, &gemm, cs, cms);
        kv.len += n;
        return cudaGetLastError() == cudaSuccess;
    };
    auto sample = [&]() -> int {
        cudaMemcpy(hl.data(), bufs.logits, m.vocab*2, cudaMemcpyDeviceToHost);
        for (int v = 0; v < m.vocab; ++v) lf[v] = __half2float(hl[v]);
        return temp > 0.0f ? sample_temperature(lf.data(), m.vocab, temp, (float)rand()/((float)RAND_MAX+1.0f))
                           : sample_greedy(lf.data(), m.vocab);
    };
    auto reset = [&]() { kv.len = 0; };
    chat_repl(tok, interactive, prompt, n_pred, eval, sample, reset);

    cudaFree(kv.K); cudaFree(kv.V); cudaFree(d_ids);
    cudaFree(bufs.hidden); cudaFree(bufs.normed); cudaFree(bufs.positions); cudaFree(bufs.logits); cudaFree(bufs.arena);
    cudaFree(s.xn); cudaFree(s.q); cudaFree(s.k); cudaFree(s.v); cudaFree(s.att); cudaFree(s.proj); cudaFree(s.gate); cudaFree(s.up);
    slotpool_destroy(&pool); gemm_destroy(&gemm); free_model(&m);
    tokenizer_free(&tok); gguf_close(&g);
    return 0;
}

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "tokenize") return cmd_tokenize(argc, argv);
    if (argc >= 2 && (std::string(argv[1]) == "run" || std::string(argv[1]) == "chat")) return cmd_run(argc, argv);
    if (argc < 4) { printf("usage: %s <model.gguf> <n_generate> <tok0> [tok1 ...]   (raw token ids)\n"
                           "       %s run <model.gguf> [-p TEXT | -it] [-n N] [--temp T] [--budget GB] [--ctx N]\n"
                           "       %s tokenize <model.gguf> <text...>\n", argv[0], argv[0], argv[0]); return 2; }
    int n_gen = atoi(argv[2]);
    std::vector<int> ids;
    for (int i = 3; i < argc; ++i) ids.push_back(atoi(argv[i]));
    int prompt_len = (int)ids.size();
    int max_T = prompt_len + n_gen;

    report_mem("start");
    GgufFile g; std::string err;
    if (!gguf_load(argv[1], &g, &err)) { printf("gguf_load: %s\n", err.c_str()); return 1; }
    int q_bits = 4;   // stream Q4_0 by default (4x smaller than fp16); SEMILLM_QBITS=8 for Q8_0
    const char* qb = getenv("SEMILLM_QBITS");
    if (qb && atoi(qb) == 8) q_bits = 8;

    // MoE models: separate path (router + experts), streaming original quant from mmap.
    if (is_moe_model(g)) {
        LoadedMoeModel mm;
        if (!load_moe_model(g, &mm, &err)) { printf("load_moe_model: %s\n", err.c_str()); return 1; }
        printf("MoE: arch=%s layers=%d dim=%d experts=%d/%d expert_ffn=%d vocab=%d | %.1f GB fp16, streamed from disk\n",
               mm.arch.c_str(), mm.n_layers, mm.cfg.dim, mm.mcfg.n_used, mm.mcfg.n_experts,
               mm.mcfg.expert_ffn, mm.vocab, mm.blob.total_elems * 2.0 / 1e9 * mm.n_layers);
        report_mem("after load");   // mmap kept alive for streaming
        double budget = 8.0;
        const char* bg = getenv("SEMILLM_BUDGET"); if (bg && atof(bg) > 0) budget = atof(bg);
        if (!moe_generate(mm, ids, n_gen, budget, &err)) { printf("moe_generate: %s\n", err.c_str()); return 1; }
        printf("generated ids:");
        for (int i = prompt_len; i < (int)ids.size(); ++i) printf(" %d", ids[i]);
        printf("\nfull sequence:"); for (int id : ids) printf(" %d", id); printf("\n");
        report_mem("after gen");
        gguf_close(&g);
        return 0;
    }

    LoadedModel m;
    if (!load_model(g, &m, q_bits, &err)) { printf("load_model: %s\n", err.c_str()); return 1; }
    gguf_close(&g);   // weights extracted; release the mmap
    const LlamaConfig& cfg = m.cfg;
    printf("model: arch=%s layers=%d dim=%d heads=%d/%d ffn=%d vocab=%d | %.1f GB Q%d streamed (vs %.1f GB fp16)\n",
           m.arch.c_str(), m.n_layers, cfg.dim, cfg.n_heads, cfg.n_kv_heads, cfg.ffn_dim,
           m.vocab, m.q8_layer_bytes / 1e9 * m.n_layers, m.q_bits, m.blob.total_elems * 2.0 * m.n_layers / 1e9);

    int qd = cfg.n_heads*cfg.head_dim, kvd = cfg.n_kv_heads*cfg.head_dim, ffn = cfg.ffn_dim;
    RunnerBufs bufs;
    cudaMalloc(&bufs.hidden, (size_t)max_T*cfg.dim*sizeof(__half));
    cudaMalloc(&bufs.normed, (size_t)max_T*cfg.dim*sizeof(__half));
    cudaMalloc(&bufs.positions, max_T*sizeof(int));
    cudaMalloc(&bufs.logits, (size_t)m.vocab*sizeof(__half));
    cudaMalloc(&bufs.arena, (size_t)m.blob.total_elems*sizeof(__half));   // fp16 dequant scratch
    LayerScratch s;
    cudaMalloc(&s.xn, (size_t)max_T*cfg.dim*sizeof(__half)); cudaMalloc(&s.q, (size_t)max_T*qd*sizeof(__half));
    cudaMalloc(&s.k, (size_t)max_T*kvd*sizeof(__half)); cudaMalloc(&s.v, (size_t)max_T*kvd*sizeof(__half));
    cudaMalloc(&s.att, (size_t)max_T*qd*sizeof(__half)); cudaMalloc(&s.proj, (size_t)max_T*cfg.dim*sizeof(__half));
    cudaMalloc(&s.gate, (size_t)max_T*ffn*sizeof(__half)); cudaMalloc(&s.up, (size_t)max_T*ffn*sizeof(__half));

    int n_slots = 4, batch_layers = 1;
    const char* slots_env = getenv("SEMILLM_SLOTS");
    if (slots_env) { int v = atoi(slots_env); if (v >= 2) n_slots = v; }
    const char* batch_env = getenv("SEMILLM_BATCH");
    if (batch_env) { int v = atoi(batch_env); if (v >= 1) batch_layers = v; }
    size_t slot_elems = ((size_t)batch_layers * m.q8_layer_bytes + 1) / 2;   // __half units covering Q8 bytes
    SlotPool pool; slotpool_create(&pool, n_slots, slot_elems);
    printf("streamed working set = %d slots x %d layers/batch x %.1f MB (Q8) = %.2f GB VRAM\n",
           n_slots, batch_layers, m.q8_layer_bytes/1e6,
           (double)n_slots*batch_layers*m.q8_layer_bytes/1e9);
    report_mem("after alloc");

    int* d_ids; cudaMalloc(&d_ids, max_T*sizeof(int));
    cudaStream_t cs, ms; cudaStreamCreate(&cs); cudaStreamCreate(&ms);
    Gemm gemm; gemm_create(&gemm);
    std::vector<__half> logits(m.vocab);
    std::vector<float> lf(m.vocab);

    // KV cache: persistent K/V per layer, so decode processes 1 new token per step
    // (not the whole growing prefix).
    KVCache kv;
    kv.n_layers = m.n_layers; kv.max_T = max_T; kv.kvd = cfg.n_kv_heads * cfg.head_dim; kv.len = 0;
    size_t kv_elems = (size_t)m.n_layers * max_T * kv.kvd;
    cudaMalloc(&kv.K, kv_elems * sizeof(__half));
    cudaMalloc(&kv.V, kv_elems * sizeof(__half));

    auto sample_next = [&]() -> int {
        cudaMemcpy(logits.data(), bufs.logits, m.vocab*sizeof(__half), cudaMemcpyDeviceToHost);
        for (int v = 0; v < m.vocab; ++v) lf[v] = __half2float(logits[v]);
        return sample_greedy(lf.data(), m.vocab);
    };

    printf("\nprefill %d prompt tokens + generate %d (KV cache)...\n", prompt_len, n_gen);
    // Prefill: whole prompt through the cache in one pass -> first token.
    cudaMemcpy(d_ids, ids.data(), prompt_len*sizeof(int), cudaMemcpyHostToDevice);
    forward_logits_cached(cfg, m.blob, m.h_layer_q8, m.q8_layer_bytes, m.q_bits, m.n_layers, batch_layers,
                          m.rw, d_ids, prompt_len, 0, kv, m.vocab, bufs, s, pool, &gemm, cs, ms);
    kv.len = prompt_len;
    int next = sample_next();
    ids.push_back(next);

    // Decode: one token per step.
    auto t0 = std::chrono::steady_clock::now();
    for (int step = 1; step < n_gen; ++step) {
        cudaMemcpy(d_ids, &next, sizeof(int), cudaMemcpyHostToDevice);
        forward_logits_cached(cfg, m.blob, m.h_layer_q8, m.q8_layer_bytes, m.q_bits, m.n_layers, batch_layers,
                              m.rw, d_ids, 1, kv.len, kv, m.vocab, bufs, s, pool, &gemm, cs, ms);
        kv.len += 1;
        next = sample_next();
        ids.push_back(next);
    }
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    int decoded = n_gen - 1;

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 1; }

    printf("generated ids:");
    for (int i = prompt_len; i < (int)ids.size(); ++i) printf(" %d", ids[i]);
    printf("\nfull sequence:");
    for (int id : ids) printf(" %d", id);
    if (decoded > 0) printf("\n%.2f tokens/s decode (%d tokens in %.2fs, KV cache)\n", decoded / secs, decoded, secs);
    report_mem("after gen");

    cudaFree(kv.K); cudaFree(kv.V);
    slotpool_destroy(&pool); gemm_destroy(&gemm); free_model(&m);
    return 0;
}

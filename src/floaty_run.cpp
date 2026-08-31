// Backend-agnostic dense-Llama runner: loads a GGUF via the portable loader and runs the
// forward pass entirely through the Backend interface, so the SAME code streams a model on
// NVIDIA/GB10 (CUDA) and Apple Silicon (Metal). Quantized layer weights are read straight
// from the mmap (Backend::wrap_host, zero-copy on unified memory); only norms + the embedding
// table are dequantized to fp16 on the CPU at load. Greedy decode.
//
// usage: floaty_run <model.gguf> <n_gen> [tok0 tok1 ...]   (raw token ids; default = BOS)
#include "backend.h"
#include "floaty_model.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <sys/mman.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

static size_t total_ram() {
#if defined(__APPLE__)
    int mib[2] = {CTL_HW, HW_MEMSIZE}; uint64_t v = 0; size_t l = sizeof(v);
    return sysctl(mib, 2, &v, &l, nullptr, 0) == 0 ? (size_t)v : 0;
#else
    long p = sysconf(_SC_PHYS_PAGES), s = sysconf(_SC_PAGE_SIZE);
    return (p > 0 && s > 0) ? (size_t)p * s : 0;
#endif
}
// Page-aligned madvise over one weight matrix's mmap bytes (prefetch / evict).
static void madv_mat(const FMat& mr, int advice) {
    if (!mr.src) return;
    size_t pg = (size_t)getpagesize();
    uintptr_t a = (uintptr_t)mr.src, end = a + mr.quant_bytes;
    uintptr_t sa = a & ~(pg - 1), ea = (end + pg - 1) & ~(pg - 1);
    madvise((void*)sa, ea - sa, advice);
}
static void madv_layer(const FLayer& ly, int advice) {
    madv_mat(ly.wq, advice); madv_mat(ly.wk, advice); madv_mat(ly.wv, advice); madv_mat(ly.wo, advice);
    madv_mat(ly.gate, advice); madv_mat(ly.up, advice); madv_mat(ly.down, advice);
}

struct Session {
    FModel* m; Backend* b; int max_T, len = 0;
    bool stream = false; int resident_K = 0, chunk = 4;   // low-RAM: stream layers >= K from disk
    int dim, qd, kvd, ffn, n_heads, n_kv_heads, head_dim, vocab;
    void *hidden, *xn, *q, *att, *proj, *gate, *up, *ye, *logits, *positions, *ids;
    void *K, *V, *embd, *fnorm;
    std::vector<void*> an, fn, wq, wk, wv, wo, wg, wu, wd;
    void* head;
};

static void* up_fp16(Backend* b, const std::vector<uint16_t>& v) {
    void* d = b->alloc(v.size() * 2); b->upload(d, v.data(), v.size() * 2); return d;
}

static void init(Session& S, FModel& m, Backend& b, int max_T) {
    S.m = &m; S.b = &b; S.max_T = max_T;
    S.dim = m.dim; S.n_heads = m.n_heads; S.n_kv_heads = m.n_kv_heads; S.head_dim = m.head_dim;
    S.qd = m.n_heads * m.head_dim; S.kvd = m.n_kv_heads * m.head_dim; S.ffn = m.ffn; S.vocab = m.vocab;
    S.hidden = b.alloc((size_t)max_T*S.dim*2); S.xn = b.alloc((size_t)max_T*S.dim*2);
    S.q = b.alloc((size_t)max_T*S.qd*2); S.att = b.alloc((size_t)max_T*S.qd*2); S.proj = b.alloc((size_t)max_T*S.dim*2);
    S.gate = b.alloc((size_t)max_T*S.ffn*2); S.up = b.alloc((size_t)max_T*S.ffn*2); S.ye = b.alloc((size_t)max_T*S.dim*2);
    S.logits = b.alloc((size_t)S.vocab*2);
    S.positions = b.alloc((size_t)max_T*4); S.ids = b.alloc((size_t)max_T*4);
    S.K = b.alloc((size_t)m.n_layers*max_T*S.kvd*2); S.V = b.alloc((size_t)m.n_layers*max_T*S.kvd*2);
    S.embd = up_fp16(&b, m.token_embd); S.fnorm = up_fp16(&b, m.final_norm);
    for (auto& ly : m.layers) {
        S.an.push_back(up_fp16(&b, ly.attn_norm)); S.fn.push_back(up_fp16(&b, ly.ffn_norm));
        S.wq.push_back(b.wrap_host(ly.wq.src, ly.wq.quant_bytes)); S.wk.push_back(b.wrap_host(ly.wk.src, ly.wk.quant_bytes));
        S.wv.push_back(b.wrap_host(ly.wv.src, ly.wv.quant_bytes)); S.wo.push_back(b.wrap_host(ly.wo.src, ly.wo.quant_bytes));
        S.wg.push_back(b.wrap_host(ly.gate.src, ly.gate.quant_bytes)); S.wu.push_back(b.wrap_host(ly.up.src, ly.up.quant_bytes));
        S.wd.push_back(b.wrap_host(ly.down.src, ly.down.quant_bytes));
    }
    S.head = b.wrap_host(m.output.src, m.output.quant_bytes);
}

// Forward n_new tokens; leaves last-token logits in S.logits. Returns nothing.
static void forward(Session& S, const int* toks, int n_new) {
    FModel& m = *S.m; Backend& b = *S.b;
    int dim = S.dim, qd = S.qd, kvd = S.kvd, ffn = S.ffn, len = S.len;
    float scale = 1.0f / sqrtf((float)S.head_dim);
    std::vector<int> pos(n_new); for (int i = 0; i < n_new; ++i) pos[i] = len + i;
    b.upload(S.ids, toks, n_new*4); b.upload(S.positions, pos.data(), n_new*4);
    b.embed(S.embd, S.ids, S.hidden, n_new, dim);
    // Prime the pipeline: start disk reads for the first chunk of streamed layers so the OS
    // is fetching them before we need them.
    if (S.stream) for (int p = S.resident_K; p < S.resident_K + S.chunk && p < m.n_layers; ++p) madv_layer(m.layers[p], MADV_WILLNEED);
    for (int L = 0; L < m.n_layers; ++L) {
        FLayer& ly = m.layers[L];
        // Pipelined prefetch: keep the disk `chunk` layers AHEAD of compute, so each layer's
        // read overlaps the GPU work of the layers before it (no per-layer stall). AirLLM-style
        // streaming but on quantized bytes and deeply overlapped.
        if (S.stream && L >= S.resident_K) {
            int pf = L + S.chunk;
            if (pf < m.n_layers && pf >= S.resident_K) madv_layer(m.layers[pf], MADV_WILLNEED);
        }
        size_t kbase = (size_t)L*S.max_T*kvd, kdst = kbase + (size_t)len*kvd;
        b.rmsnorm(S.hidden, S.an[L], S.xn, n_new, dim, m.eps);
        b.fused_gemv(S.wq[L], S.xn, S.q, n_new, qd, dim, ly.wq.type);
        b.fused_gemv(S.wk[L], S.xn, S.K, n_new, kvd, dim, ly.wk.type, kdst);
        b.fused_gemv(S.wv[L], S.xn, S.V, n_new, kvd, dim, ly.wv.type, kdst);
        b.rope(S.q, S.positions, n_new, S.n_heads, S.head_dim, m.rope_base);
        b.rope(S.K, S.positions, n_new, S.n_kv_heads, S.head_dim, m.rope_base, kdst);
        b.attention(S.q, S.K, kbase, S.V, kbase, S.att, n_new, len, S.n_heads, S.n_kv_heads, S.head_dim, scale);
        b.fused_gemv(S.wo[L], S.att, S.proj, n_new, dim, qd, ly.wo.type);
        b.residual_add(S.hidden, S.proj, n_new*dim);
        b.rmsnorm(S.hidden, S.fn[L], S.xn, n_new, dim, m.eps);
        b.fused_gemv(S.wg[L], S.xn, S.gate, n_new, ffn, dim, ly.gate.type);
        b.fused_gemv(S.wu[L], S.xn, S.up, n_new, ffn, dim, ly.up.type);
        b.silu_mul(S.gate, S.up, n_new*ffn);
        b.fused_gemv(S.wd[L], S.gate, S.ye, n_new, dim, ffn, ly.down.type);
        b.residual_add(S.hidden, S.ye, n_new*dim);
        // At each chunk boundary: sync (the streamed layers in this chunk are done being read
        // by the GPU), then evict the chunk's pages. Footprint stays ~ resident_K + 2*chunk
        // (in-flight + prefetched), while the NEXT chunk's disk reads already overlapped this
        // chunk's compute. Bigger chunk = more overlap, larger footprint (FLOATY_STREAM_CHUNK).
        if (S.stream && L >= S.resident_K &&
            ((L - S.resident_K + 1) % S.chunk == 0 || L == m.n_layers - 1)) {
            b.sync();
            int lo = S.resident_K + ((L - S.resident_K) / S.chunk) * S.chunk;
            for (int e = lo; e <= L; ++e) madv_layer(m.layers[e], MADV_DONTNEED);
        }
    }
    b.rmsnorm(S.hidden, S.fnorm, S.xn, n_new, dim, m.eps);
    b.fused_gemv(S.head, S.xn, S.logits, 1, S.vocab, dim, m.output.type, 0, (size_t)(n_new-1)*dim);
    S.len += n_new;
}

// SentencePiece-ish detok of one vocab piece.
static std::string detok(const MetaValue* toks, int id) {
    if (!toks || id < 0 || id >= (int)toks->strs.size()) return "";
    std::string t = toks->strs[id], out;
    for (size_t i = 0; i < t.size();) {
        if (i + 3 <= t.size() && (uint8_t)t[i] == 0xE2 && (uint8_t)t[i+1] == 0x96 && (uint8_t)t[i+2] == 0x81) { out += ' '; i += 3; }
        else if (t.size() == 6 && t[0]=='<' && t[1]=='0' && t[2]=='x') { out += (char)strtol(t.c_str()+3, nullptr, 16); i = t.size(); }
        else { out += t[i]; ++i; }
    }
    return out;
}

int main(int argc, char** argv) {
    if (argc < 3) { printf("usage: %s <model.gguf> <n_gen> [tok0 tok1 ...]\n", argv[0]); return 2; }
    const char* path = argv[1]; int n_gen = atoi(argv[2]);
    FModel m; std::string err;
    if (!floaty_load(path, &m, &err)) { printf("load: %s\n", err.c_str()); return 1; }
    Backend* b =
#ifdef FLOATY_METAL
        make_metal_backend();
#else
        make_cuda_backend();
#endif
    if (!b) { printf("no backend\n"); return 1; }
    printf("FloatyLLM · %s · %s · %d layers · dim %d · vocab %d\n", b->name(), m.arch.c_str(), m.n_layers, m.dim, m.vocab);

    std::vector<int> ids;
    for (int i = 3; i < argc; ++i) ids.push_back(atoi(argv[i]));
    if (ids.empty()) { uint32_t bos; ids.push_back(gguf_get_u32(m.g, "tokenizer.ggml.bos_token_id", &bos) ? (int)bos : 1); }

    Session S; init(S, m, *b, ids.size() + n_gen + 8);
    // Decide resident vs stream from the model size and a RAM budget (auto = 70% of RAM;
    // override with FLOATY_RAM_GB). Stream mode keeps the first K layers resident and pages
    // the rest from disk per token -> small footprint, runs in low RAM (AirLLM regime) but
    // on quantized bytes + with K layers cached -> faster than AirLLM.
    size_t model_bytes = 0;
    for (auto& ly : m.layers) model_bytes += ly.wq.quant_bytes + ly.wk.quant_bytes + ly.wv.quant_bytes
        + ly.wo.quant_bytes + ly.gate.quant_bytes + ly.up.quant_bytes + ly.down.quant_bytes;
    size_t ram = total_ram(); size_t budget = ram ? ram * 7 / 10 : (size_t)-1;
    if (const char* e = getenv("FLOATY_RAM_GB")) budget = (size_t)(atof(e) * 1e9);
    size_t lb = model_bytes / (m.n_layers > 0 ? m.n_layers : 1);
    if (const char* e = getenv("FLOATY_STREAM_CHUNK")) { int c = atoi(e); if (c >= 1) S.chunk = c; }
    if (model_bytes > budget) {
        S.stream = true; S.resident_K = (int)(budget / (lb ? lb : 1));
        if (S.resident_K < 0) S.resident_K = 0; if (S.resident_K > m.n_layers) S.resident_K = m.n_layers;
        for (int L = 0; L < S.resident_K; ++L) madv_layer(m.layers[L], MADV_WILLNEED);   // pin resident set
        printf("stream mode: model %.2f GB > budget %.2f GB -> %d/%d layers resident, rest streamed (chunk %d, prefetch-overlapped)\n",
               model_bytes / 1e9, budget / 1e9, S.resident_K, m.n_layers, S.chunk);
    } else {
        printf("resident: model %.2f GB fits budget %.2f GB\n", model_bytes / 1e9, budget / 1e9);
    }
    auto it = m.g.meta.find("tokenizer.ggml.tokens");
    const MetaValue* vtoks = it != m.g.meta.end() ? &it->second : nullptr;

    forward(S, ids.data(), (int)ids.size()); b->sync();
    int next = b->argmax(S.logits, S.vocab);
    auto t0 = std::chrono::steady_clock::now();
    std::string outtext;
    for (int i = 0; i < n_gen; ++i) {
        std::string piece = detok(vtoks, next); outtext += piece;
        fputs(piece.c_str(), stdout); fflush(stdout);
        forward(S, &next, 1); b->sync();
        next = b->argmax(S.logits, S.vocab);
    }
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("\n[%d tok, %.2f tok/s]\n", n_gen, n_gen / (secs > 0 ? secs : 1));
    floaty_free(&m);
    return 0;
}

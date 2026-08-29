// MoE model: stream the ORIGINAL quantized weights from the mmap'd GGUF, dequant +
// transpose on the GPU per layer (no host requant, no pinned whole-model copy). Only
// the OS working set (~K layers) stays in RAM. PLAN sec 16.
#pragma once
#include "weights.h"
#include "moe.h"
#include "stream.h"
#include <string>
#include <vector>

// Per-layer packed blob (fp16 element offsets): attn + router + n_experts experts.
struct MoeLayerBlob {
    size_t off_attn_norm, off_wq, off_wk, off_wv, off_wo, off_ffn_norm, off_router, off_experts;
    size_t expert_stride, off_egate, off_eup, off_edown;
    size_t total_elems;
};

// One weight matrix: where its quantized bytes live (mmap ptr) + how to place it.
struct MatRef {
    const uint8_t* src;   // mmap pointer to the quantized data
    uint32_t type;        // ggml type
    int out, in;          // [out,in] tensor dims
    size_t n_elems;       // out*in (or dim for a norm)
    size_t quant_bytes;   // bytes to copy
    size_t arena_off;     // fp16 element offset in the layer arena
    bool is_norm;         // norms are 1D (no transpose)
    int expert;           // expert id if an expert matrix, else -1 (attn/router/norm)
};

struct LoadedMoeModel {
    LlamaConfig cfg;
    MoeConfig mcfg;
    int n_layers = 0, vocab = 0;
    std::string arch;
    MoeLayerBlob blob;
    const GgufFile* g = nullptr;                    // mmap, kept alive by the caller
    std::vector<std::vector<MatRef>> layers;        // per-layer matrix list
    size_t max_quant_bytes = 0, max_elems = 0;      // largest single matrix (staging sizes)
    RunnerWeights rw{};
};

bool is_moe_model(const GgufFile& g);
bool load_moe_model(const GgufFile& g, LoadedMoeModel* out, std::string* err);
void free_moe_model(LoadedMoeModel* m);

// Greedy generate. budget_gb caps the resident streaming VRAM; K resident layers is
// derived from it and the per-layer size (dynamic). Appends generated ids to *ids.
bool moe_generate(LoadedMoeModel& m, std::vector<int>& ids, int n_gen,
                  double budget_gb, std::string* err);

// Hot-expert VRAM cache (PLAN sec 16): keep recently-used experts resident (dequantized
// fp16). PARTITIONED PER LAYER — each layer owns `per_layer` slots with its own LRU, so
// one layer's sweep never evicts another's (no sequential-sweep thrash). per_layer spans
// n_used (minimal) -> n_experts (whole model resident). A routing hit skips the DMA +
// dequant + transpose. Per-slot events guard slot reuse across the copy/compute streams.
struct ExpertCache {
    int capacity = 0;                  // total slots (per-layer: n_layers*per_layer; global: budget-derived)
    int per_layer = 0;                 // resident experts per layer (MoE mode)
    bool global = false;               // dense mode: one global LRU keyed by layer, stream the rest
    int n_experts = 0;
    size_t stride = 0;                 // fp16 elems per expert (gate|up|down = 3*dim*ef)
    __half* pool = nullptr;            // capacity * stride
    std::vector<int> slot_key;         // slot -> key (per-layer: expert id; global: layer). -1 empty
    std::vector<uint64_t> slot_lru;    // slot -> last-use tick
    std::vector<cudaEvent_t> slot_evt; // recorded on compute stream after a slot is read
    uint64_t tick = 0;
    long hits = 0, misses = 0;
};

// --- Persistent session: load once, evaluate many times with the KV cache kept, so a
// REPL/chat can keep conversation context across turns (PLAN sec 6). ---
struct MoeSession {
    LoadedMoeModel* m = nullptr;
    int max_T = 0, len = 0;                 // len = tokens already in the KV cache
    double budget_gb = 0;
    __half *hidden = nullptr, *normed = nullptr, *arena = nullptr, *d_deq = nullptr, *logits_d = nullptr;
    uint8_t* d_qstage = nullptr; int* positions = nullptr; int* d_ids = nullptr; int* d_arg = nullptr;
    LayerScratch s{}; MoeScratch ms{}; KVCache kv{};
    std::vector<const __half*> wg, wu, wd; MoeLayerWeights w{};   // per-layer active-expert ptrs
    ExpertCache cache;
    // Device-resident QUANTIZED weight cache for the dense fused-decode path: GB10 host
    // mmap reads cap ~160 GB/s vs ~220 GB/s from GPU-local memory, so hold as many whole
    // layers as the budget allows in device DRAM. qsrc[L][i] = effective src of mat i of
    // layer L (device pool if resident, else the host mmap pointer).
    uint8_t* qpool = nullptr;
    std::vector<std::vector<const uint8_t*>> qsrc;
    int qres_layers = 0;
    cudaStream_t cm = nullptr, cs = nullptr; std::vector<cudaEvent_t> ev; Gemm gemm{};
    std::vector<__half> hl; std::vector<float> lf;
    long exp_streamed = 0, exp_total = 0;   // instrumentation across all evals
};

bool moe_session_init(LoadedMoeModel& m, int max_T, double budget_gb, MoeSession* S, std::string* err);
// Evaluate n_new host-side token ids at the current KV position; updates logits for the
// LAST token and advances len by n_new. Returns false on a CUDA error.
bool moe_session_eval(MoeSession& S, const int* ids, int n_new, std::string* err);
// Sample from the last logits (temp <= 0 => greedy; rand01 in [0,1) for temperature).
int  moe_session_sample(MoeSession& S, float temp, float rand01);
void moe_session_reset(MoeSession& S);      // clear conversation (len = 0)
void moe_session_free(MoeSession* S);       // frees session buffers, NOT the model

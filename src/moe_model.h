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

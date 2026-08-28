// MoE model loading (3D expert tensors) + config + generation. Mixtral-style
// (dense Llama attention + MoE MLP). PLAN sec 16.
#pragma once
#include "weights.h"
#include "moe.h"
#include "stream.h"
#include <string>
#include <vector>

// Per-layer packed blob (fp16 element offsets): attn + router + n_experts experts.
struct MoeLayerBlob {
    size_t off_attn_norm, off_wq, off_wk, off_wv, off_wo, off_ffn_norm, off_router, off_experts;
    size_t expert_stride;                 // elements per expert (gate+up+down)
    size_t off_egate, off_eup, off_edown; // offsets within one expert
    size_t total_elems;
};

struct LoadedMoeModel {
    LlamaConfig cfg;
    MoeConfig mcfg;
    int n_layers = 0, vocab = 0, q_bits = 8;
    std::string arch;
    MoeLayerBlob blob;
    uint8_t* h_layer_q = nullptr;   // pinned Q-quant per-layer blobs
    size_t q_layer_bytes = 0;
    RunnerWeights rw{};
};

// True if the GGUF declares experts ({arch}.expert_count > 1).
bool is_moe_model(const GgufFile& g);

bool load_moe_model(const GgufFile& g, LoadedMoeModel* out, int q_bits, std::string* err);
void free_moe_model(LoadedMoeModel* m);

// Greedy generate n_gen tokens from prompt ids (prefill + KV-cache decode). Appends
// generated ids to *ids. Prints per-token progress + tokens/s.
bool moe_generate(LoadedMoeModel& m, std::vector<int>& ids, int n_gen,
                  int n_slots, int batch_layers, std::string* err);

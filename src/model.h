// Model config + per-layer tensor binding, read from a parsed GGUF using the
// llama.cpp metadata + tensor naming conventions. Host-only. PLAN sec 8.
#pragma once
#include "loader.h"
#include <string>

struct ModelConfig {
    std::string arch;
    int n_layers = 0, dim = 0, n_heads = 0, n_kv_heads = 0, head_dim = 0, ffn_dim = 0;
    float eps = 1e-5f, rope_base = 10000.0f;
};

// The nine weight tensors of one decoder layer (pointers into the GgufFile).
struct LayerTensors {
    const TensorInfo *attn_norm, *wq, *wk, *wv, *wo;
    const TensorInfo *ffn_norm, *wgate, *wup, *wdown;
};

// Read config from {arch}.* metadata keys. Returns false + *err on missing keys.
bool read_config(const GgufFile& g, ModelConfig* out, std::string* err);

// Resolve one layer's tensors by name (blk.{i}.*). Returns false + *err if any missing.
bool get_layer_tensors(const GgufFile& g, int layer, LayerTensors* out, std::string* err);

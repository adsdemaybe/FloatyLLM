// Real-model weight loader: GGUF tensors -> fp16, dequantized + transposed into
// the streaming blob layout + non-layer weights on device. PLAN sec 8.
#pragma once
#include "loader.h"
#include "model.h"
#include "stream.h"
#include "runner.h"
#include <string>

struct LoadedModel {
    LlamaConfig cfg;      // per-layer compute config (for forward_logits)
    int n_layers = 0;     // from ModelConfig.block_count
    int vocab = 0;
    std::string arch;
    LayerBlob blob;                       // fp16 element offsets (arena layout)
    // Weights are re-quantized to Q8_0 for streaming (half the RAM + transfer of
    // fp16). Each layer = (blob.total_elems/32) Q8_0 blocks of 34 bytes.
    uint8_t* h_layer_q8 = nullptr;        // pinned, n_layers * q8_layer_bytes
    size_t q8_layer_bytes = 0;
    RunnerWeights rw{};                   // device token_embd / final_norm / output
};

// Dequant + transpose all weights from a parsed GGUF into a LoadedModel.
// Supports F32/F16/Q8_0 tensors (K-quants added later). Returns false + *err.
bool load_model(const GgufFile& g, LoadedModel* out, std::string* err);
void free_model(LoadedModel* m);

// Streaming scheduler: ring of VRAM weight slots + copy/compute pipeline that
// drives layer_forward over host-resident weights. The core of SemiLLM - run a
// model whose weights exceed VRAM by streaming layers. PLAN sec 6. fp16 weights
// for now (quantized streaming + dequant is a later refinement).
#pragma once
#include "layer.h"

// Byte/element offsets of the 9 weight tensors within a packed per-layer blob.
struct LayerBlob {
    size_t off_attn_norm, off_wq, off_wk, off_wv, off_wo;
    size_t off_ffn_norm, off_wgate, off_wup, off_wdown;
    size_t total_elems;   // fp16 elements per layer
};

LayerBlob layer_blob_layout(const LlamaConfig& cfg);

// Ring of VRAM weight slots (one packed layer each) + per-slot events.
struct SlotPool {
    int n_slots = 0;
    size_t slot_elems = 0;
    __half** d_slots = nullptr;
    cudaEvent_t* copy_done = nullptr;
    cudaEvent_t* compute_done = nullptr;
};

void slotpool_create(SlotPool* p, int n_slots, size_t slot_elems);
void slotpool_destroy(SlotPool* p);

// Build a LayerWeights view into a slot's packed blob.
LayerWeights weights_from_blob(const __half* base, const LayerBlob& b);

// Stream n_layers of packed fp16 weights (h_weights: pinned, layer-major,
// n_layers * blob.total_elems) through layer_forward, updating hidden in place.
// Copy runs on copy_stream, compute on compute_stream; execute-on-completion.
void stream_forward(const LlamaConfig& cfg, const LayerBlob& blob,
                    const __half* h_weights, int n_layers,
                    __half* hidden, const int* d_positions, int n_tokens,
                    LayerScratch& scratch, SlotPool& pool, Gemm* gemm,
                    cudaStream_t copy_stream, cudaStream_t compute_stream);

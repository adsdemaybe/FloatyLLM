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
// batch_layers = layers copied per DMA: a batch is contiguous in h_weights, so
// each prefetch is ONE big cudaMemcpyAsync of batch_layers layers -> maximizes
// PCIe transfer size while the previous batch computes. pool slots must be sized
// batch_layers * blob.total_elems. Copy on copy_stream, compute on compute_stream.
void stream_forward(const LlamaConfig& cfg, const LayerBlob& blob,
                    const __half* h_weights, int n_layers, int batch_layers,
                    __half* hidden, const int* d_positions, int n_tokens,
                    LayerScratch& scratch, SlotPool& pool, Gemm* gemm,
                    cudaStream_t copy_stream, cudaStream_t compute_stream);

// Quantized streaming: h_q8 holds per-layer Q8_0 blobs (q8_layer_bytes each). A
// batch of Q8_0 layers is copied in ONE DMA (half the bytes of fp16), then each
// layer is dequantized on the GPU into `arena` (fp16, blob.total_elems) before
// layer_forward. pool slots hold batch_layers * q8_layer_bytes bytes.
void stream_forward_q8(const LlamaConfig& cfg, const LayerBlob& blob,
                       const uint8_t* h_q8, size_t q8_layer_bytes, int n_layers,
                       int batch_layers, __half* arena, __half* hidden,
                       const int* d_positions, int n_tokens, LayerScratch& scratch,
                       SlotPool& pool, Gemm* gemm,
                       cudaStream_t copy_stream, cudaStream_t compute_stream);

// Quantized streaming with KV cache: processes n_new tokens through every streamed
// layer, using layer_forward_cached (append K/V at len_before, attend the cache).
void stream_forward_q8_cached(const LlamaConfig& cfg, const LayerBlob& blob,
                              const uint8_t* h_q8, size_t q8_layer_bytes, int n_layers,
                              int batch_layers, __half* arena, __half* hidden,
                              const int* d_positions, int n_new, int len_before, KVCache& kv,
                              LayerScratch& scratch, SlotPool& pool, Gemm* gemm,
                              cudaStream_t copy_stream, cudaStream_t compute_stream);

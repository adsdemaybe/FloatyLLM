// Dense runner: token ids -> embed -> streamed layers -> final norm -> LM head
// -> logits for the last token. Ties the whole forward together. PLAN sec 3/6.
#pragma once
#include "stream.h"

// Non-layer weights (the layer weights are streamed via stream_forward).
struct RunnerWeights {
    const __half* token_embd;   // [vocab, dim] row-major (row = token id)
    const __half* final_norm;   // [dim]
    const __half* output;       // [vocab, dim] row-major (LM head; row = vocab entry)
};

// Caller-allocated device buffers for one forward.
struct RunnerBufs {
    __half* hidden;   // [T, dim]
    __half* normed;   // [T, dim]
    int*    positions;// [T]  (0..T-1)
    __half* logits;   // [vocab]  (for the last token)
    __half* arena;    // [blob.total_elems]  (fp16 dequant scratch, quantized path)
};

// Embed token ids (device) into hidden [T, dim].
void embed_tokens(const __half* token_embd, const int* d_ids, __half* d_hidden,
                  int T, int dim, cudaStream_t stream);

// Full forward over n_layers streamed layers -> logits for the last token.
// d_ids: device token ids [T]. h_layer_weights: pinned, layer-major packed blobs.
void forward_logits(const LlamaConfig& cfg, const LayerBlob& blob,
                    const __half* h_layer_weights, int n_layers, int batch_layers,
                    const RunnerWeights& rw, const int* d_ids, int T, int vocab,
                    RunnerBufs& bufs, LayerScratch& scratch, SlotPool& pool, Gemm* gemm,
                    cudaStream_t copy_stream, cudaStream_t compute_stream);

// Quantized-streaming forward: streams Q8_0 layer blobs, dequant on GPU (bufs.arena).
void forward_logits_q8(const LlamaConfig& cfg, const LayerBlob& blob,
                       const uint8_t* h_q8, size_t q8_layer_bytes, int n_layers,
                       int batch_layers, const RunnerWeights& rw, const int* d_ids,
                       int T, int vocab, RunnerBufs& bufs, LayerScratch& scratch,
                       SlotPool& pool, Gemm* gemm,
                       cudaStream_t copy_stream, cudaStream_t compute_stream);

// Cached quantized forward: process n_new tokens with a KV cache (prefill n_new=prompt,
// decode n_new=1). Logits are for the last of the n_new tokens.
void forward_logits_cached(const LlamaConfig& cfg, const LayerBlob& blob,
                           const uint8_t* h_q8, size_t q8_layer_bytes, int n_layers,
                           int batch_layers, const RunnerWeights& rw, const int* d_ids,
                           int n_new, int len_before, KVCache& kv, int vocab,
                           RunnerBufs& bufs, LayerScratch& scratch, SlotPool& pool, Gemm* gemm,
                           cudaStream_t copy_stream, cudaStream_t compute_stream);

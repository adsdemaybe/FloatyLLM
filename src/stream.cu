// Streaming scheduler implementation.
#include "stream.h"
#include "dequant.h"

LayerBlob layer_blob_layout(const LlamaConfig& cfg) {
    size_t dim = cfg.dim;
    size_t qd = (size_t)cfg.n_heads * cfg.head_dim;
    size_t kvd = (size_t)cfg.n_kv_heads * cfg.head_dim;
    size_t ffn = cfg.ffn_dim;

    LayerBlob b;
    b.off_attn_norm = 0;
    b.off_wq       = b.off_attn_norm + dim;
    b.off_wk       = b.off_wq + dim * qd;
    b.off_wv       = b.off_wk + dim * kvd;
    b.off_wo       = b.off_wv + dim * kvd;
    b.off_ffn_norm = b.off_wo + qd * dim;
    b.off_wgate    = b.off_ffn_norm + dim;
    b.off_wup      = b.off_wgate + dim * ffn;
    b.off_wdown    = b.off_wup + dim * ffn;
    b.total_elems  = b.off_wdown + ffn * dim;
    return b;
}

LayerWeights weights_from_blob(const __half* base, const LayerBlob& b) {
    LayerWeights w;
    w.attn_norm = base + b.off_attn_norm;
    w.wq        = base + b.off_wq;
    w.wk        = base + b.off_wk;
    w.wv        = base + b.off_wv;
    w.wo        = base + b.off_wo;
    w.ffn_norm  = base + b.off_ffn_norm;
    w.wgate     = base + b.off_wgate;
    w.wup       = base + b.off_wup;
    w.wdown     = base + b.off_wdown;
    return w;
}

void slotpool_create(SlotPool* p, int n_slots, size_t slot_elems) {
    p->n_slots = n_slots;
    p->slot_elems = slot_elems;
    p->d_slots = new __half*[n_slots];
    p->copy_done = new cudaEvent_t[n_slots];
    p->compute_done = new cudaEvent_t[n_slots];
    for (int i = 0; i < n_slots; ++i) {
        cudaMalloc(&p->d_slots[i], slot_elems * sizeof(__half));
        cudaEventCreate(&p->copy_done[i]);
        cudaEventCreate(&p->compute_done[i]);
    }
}

void slotpool_destroy(SlotPool* p) {
    for (int i = 0; i < p->n_slots; ++i) {
        cudaFree(p->d_slots[i]);
        cudaEventDestroy(p->copy_done[i]);
        cudaEventDestroy(p->compute_done[i]);
    }
    delete[] p->d_slots;
    delete[] p->copy_done;
    delete[] p->compute_done;
    p->n_slots = 0;
}

void stream_forward(const LlamaConfig& cfg, const LayerBlob& blob,
                    const __half* h_weights, int n_layers, int batch_layers,
                    __half* hidden, const int* d_positions, int n_tokens,
                    LayerScratch& scratch, SlotPool& pool, Gemm* gemm,
                    cudaStream_t copy_stream, cudaStream_t compute_stream) {
    if (batch_layers < 1) batch_layers = 1;
    int N = pool.n_slots;
    int n_batches = (n_layers + batch_layers - 1) / batch_layers;

    // prefetch a whole batch of contiguous layers in ONE big DMA (max PCIe size).
    auto prefetch_batch = [&](int b) {
        if (b >= n_batches) return;
        int s = b % N;
        int first = b * batch_layers;
        int cnt = batch_layers;
        if (first + cnt > n_layers) cnt = n_layers - first;
        cudaStreamWaitEvent(copy_stream, pool.compute_done[s], 0);   // slot (batch) free?
        cudaMemcpyAsync(pool.d_slots[s], h_weights + (size_t)first * blob.total_elems,
                        (size_t)cnt * blob.total_elems * sizeof(__half),
                        cudaMemcpyHostToDevice, copy_stream);        // one transfer, B layers
        cudaEventRecord(pool.copy_done[s], copy_stream);
    };

    prefetch_batch(0);
    for (int b = 0; b < n_batches; ++b) {
        prefetch_batch(b + 1);                                       // next batch while this computes
        int s = b % N;
        int first = b * batch_layers;
        int cnt = (first + batch_layers > n_layers) ? (n_layers - first) : batch_layers;
        cudaStreamWaitEvent(compute_stream, pool.copy_done[s], 0);
        for (int l = 0; l < cnt; ++l) {
            LayerWeights w = weights_from_blob(pool.d_slots[s] + (size_t)l * blob.total_elems, blob);
            layer_forward(cfg, w, hidden, d_positions, n_tokens, scratch, gemm, compute_stream);
        }
        cudaEventRecord(pool.compute_done[s], compute_stream);
    }
    cudaStreamSynchronize(compute_stream);
}

void stream_forward_q8(const LlamaConfig& cfg, const LayerBlob& blob,
                       const uint8_t* h_q8, size_t q8_layer_bytes, int n_layers,
                       int batch_layers, __half* arena, __half* hidden,
                       const int* d_positions, int n_tokens, LayerScratch& scratch,
                       SlotPool& pool, Gemm* gemm,
                       cudaStream_t copy_stream, cudaStream_t compute_stream) {
    if (batch_layers < 1) batch_layers = 1;
    int N = pool.n_slots;
    int n_batches = (n_layers + batch_layers - 1) / batch_layers;
    int blocks_per_layer = (int)(blob.total_elems / 32);

    auto prefetch_batch = [&](int b) {
        if (b >= n_batches) return;
        int s = b % N;
        int first = b * batch_layers;
        int cnt = (first + batch_layers > n_layers) ? (n_layers - first) : batch_layers;
        cudaStreamWaitEvent(copy_stream, pool.compute_done[s], 0);
        cudaMemcpyAsync((uint8_t*)pool.d_slots[s], h_q8 + (size_t)first * q8_layer_bytes,
                        (size_t)cnt * q8_layer_bytes, cudaMemcpyHostToDevice, copy_stream);
        cudaEventRecord(pool.copy_done[s], copy_stream);
    };

    prefetch_batch(0);
    for (int b = 0; b < n_batches; ++b) {
        prefetch_batch(b + 1);
        int s = b % N;
        int first = b * batch_layers;
        int cnt = (first + batch_layers > n_layers) ? (n_layers - first) : batch_layers;
        cudaStreamWaitEvent(compute_stream, pool.copy_done[s], 0);
        const uint8_t* slot = (const uint8_t*)pool.d_slots[s];
        for (int l = 0; l < cnt; ++l) {
            const BlockQ80* q8 = (const BlockQ80*)(slot + (size_t)l * q8_layer_bytes);
            dequant_q8_0(q8, arena, blocks_per_layer, compute_stream);   // GPU dequant -> fp16 arena
            LayerWeights w = weights_from_blob(arena, blob);
            layer_forward(cfg, w, hidden, d_positions, n_tokens, scratch, gemm, compute_stream);
        }
        cudaEventRecord(pool.compute_done[s], compute_stream);
    }
    cudaStreamSynchronize(compute_stream);
}

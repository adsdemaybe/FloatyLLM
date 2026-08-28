// Streaming scheduler implementation.
#include "stream.h"

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
                    const __half* h_weights, int n_layers,
                    __half* hidden, const int* d_positions, int n_tokens,
                    LayerScratch& scratch, SlotPool& pool, Gemm* gemm,
                    cudaStream_t copy_stream, cudaStream_t compute_stream) {
    int N = pool.n_slots;

    // prefetch(L): copy layer L's blob into its ring slot on the copy stream,
    // after the slot's previous occupant finished compute (device-side wait).
    auto prefetch = [&](int L) {
        if (L >= n_layers) return;
        int s = L % N;
        cudaStreamWaitEvent(copy_stream, pool.compute_done[s], 0);
        cudaMemcpyAsync(pool.d_slots[s], h_weights + (size_t)L * blob.total_elems,
                        blob.total_elems * sizeof(__half), cudaMemcpyHostToDevice, copy_stream);
        cudaEventRecord(pool.copy_done[s], copy_stream);
    };

    prefetch(0);
    for (int L = 0; L < n_layers; ++L) {
        prefetch(L + 1);                                   // keep copy engine ahead
        int s = L % N;
        cudaStreamWaitEvent(compute_stream, pool.copy_done[s], 0);
        LayerWeights w = weights_from_blob(pool.d_slots[s], blob);
        layer_forward(cfg, w, hidden, d_positions, n_tokens, scratch, gemm, compute_stream);
        cudaEventRecord(pool.compute_done[s], compute_stream);
    }
    cudaStreamSynchronize(compute_stream);
}

// Dense runner implementation.
#include "runner.h"
#include "rmsnorm.h"

__global__ void embed_kernel(const __half* emb, const int* ids, __half* out,
                             int dim) {
    int t = blockIdx.x;
    int id = ids[t];
    const __half* row = emb + (size_t)id * dim;
    __half* o = out + (size_t)t * dim;
    for (int j = threadIdx.x; j < dim; j += blockDim.x) o[j] = row[j];
}

__global__ void iota_kernel(int* p, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = i;
}

void embed_tokens(const __half* token_embd, const int* d_ids, __half* d_hidden,
                  int T, int dim, cudaStream_t stream) {
    if (T <= 0 || dim <= 0) return;
    embed_kernel<<<T, 256, 0, stream>>>(token_embd, d_ids, d_hidden, dim);
}

void forward_logits(const LlamaConfig& cfg, const LayerBlob& blob,
                    const __half* h_layer_weights, int n_layers, int batch_layers,
                    const RunnerWeights& rw, const int* d_ids, int T, int vocab,
                    RunnerBufs& bufs, LayerScratch& scratch, SlotPool& pool, Gemm* gemm,
                    cudaStream_t copy_stream, cudaStream_t compute_stream) {
    // positions = 0..T-1
    iota_kernel<<<(T + 255) / 256, 256, 0, compute_stream>>>(bufs.positions, T);

    // embed
    embed_tokens(rw.token_embd, d_ids, bufs.hidden, T, cfg.dim, compute_stream);

    // streamed decoder layers (stream_forward syncs compute_stream at the end)
    stream_forward(cfg, blob, h_layer_weights, n_layers, batch_layers, bufs.hidden,
                   bufs.positions, T, scratch, pool, gemm, copy_stream, compute_stream);

    // final RMSNorm
    rmsnorm(bufs.hidden, rw.final_norm, bufs.normed, T, cfg.dim, cfg.eps, compute_stream);

    // LM head for the last token: logits[vocab] = output[vocab,dim] @ normed_last[dim]
    const __half* last = bufs.normed + (size_t)(T - 1) * cfg.dim;
    gemm_rowmajor(gemm, rw.output, last, bufs.logits, vocab, 1, cfg.dim, compute_stream);
    cudaStreamSynchronize(compute_stream);
}

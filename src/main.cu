// SemiLLM runtime entry point + build roadmap.
// Lays out the full inference pipeline (see PLAN.md sections 3, 6). Each stage is
// tagged [BUILT] (kernel exists + tested) or [TODO] (not yet implemented), with the
// file that owns it. main() below is a minimal real body; the pipeline it will drive
// is described in order so the remaining work is explicit.
//
// =============================================================================
// STAGE 0 - OFFLINE PACKER                                            [TODO]
//   packer/ (Python). fp32 model -> per-layer sensitivity -> bit allocation ->
//   mixed-quant weight file in VRAM-native layout + manifest (layer -> type).
//   Not part of this binary.
//
// STAGE 1 - STARTUP                                                   [TODO]
//   - loader:  mmap quant file, load whole model into pinned host RAM,
//              parse manifest.                                  src/loader.*  [TODO]
//   - vram:    alloc regions ONCE - resident cache (K quant layers), ring
//              (3-4 quant slots), one fp16 dequant arena, KV cache, cuBLAS
//              workspace.                                       src/slot_pool.* [TODO]
//   - resident: auto-size K to free VRAM, copy K layers to GPU.  src/slot_pool.* [TODO]
//   - tokenizer: load vocab from the file.                       src/loader.*   [TODO]
//
// STAGE 2 - PER-TOKEN LOOP                                            [partial]
//   embed token -> hidden [T, dim]                                            [TODO]
//   for L in 0..n_layers-1:
//       prefetch(L + ahead): H2D copy next quant layer -> ring   src/scheduler.* [TODO]
//       wait copy_done (device event)                            src/scheduler.* [TODO]
//       dequant layer L -> fp16 arena                            src/dequant.cu  [BUILT]
//       layer_forward(cfg, weights, hidden, pos, T, scratch)     src/layer.cu    [BUILT]
//           rmsnorm                                              src/rmsnorm.cu  [BUILT]
//           QKV / O / gate / up / down GEMMs                     src/gemm.cu     [BUILT]
//           RoPE on Q,K                                          src/rope.cu     [BUILT]
//           attention (causal, GQA)                              src/attention.cu[BUILT]
//           softmax (inside attention path)                      src/softmax.cu  [BUILT]
//           silu / mul / residual                                src/elementwise.cu [BUILT]
//       record compute_done (frees ring slot)                    src/scheduler.* [TODO]
//   final rmsnorm -> LM head GEMM -> logits                                     [TODO]
//   sample next token, append, grow KV cache                                   [TODO]
//
// STAGE 3 - PIPELINE OVERLAP                                          [TODO]
//   copy stream || compute stream, per-slot cudaEvents; execute-on-completion.
//   src/scheduler.* [TODO]. Optimizations (matrix-granularity, fused MMQ GEMM,
//   FlashAttention, GPUDirect Storage) come after correctness (PLAN sec 11-12).
// =============================================================================

#include <cstdio>
#include <cuda_runtime.h>

// Built compute path (compiles today; wired once the loader/scheduler exist):
#include "layer.h"
#include "dequant.h"

int main() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    printf("SemiLLM. CUDA devices visible: %d (%s)\n", count, cudaGetErrorString(err));

    // BUILT so far: dequant, rmsnorm, rope, attention, softmax, elementwise, gemm,
    //               and the full single-layer forward (layer_forward).
    // TODO next:    GGUF loader -> feed real Q8_0 weights -> match llama.cpp layer-0,
    //               then slot_pool + scheduler for the streaming pipeline.
    printf("Compute path built through layer_forward. Loader + scheduler are next.\n");
    return 0;
}

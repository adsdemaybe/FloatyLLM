// Fused dequant-GEMV: y[m,n_out] = x[m,n_in] * dequant(W[n_out,n_in]), reading the
// weight in its NATIVE quantized [out,in] layout and dequantizing in-register. Removes
// the dequant + transpose + fp16 arena seam that dominates decode on unified memory
// (PLAN sec 11, MMQ appendix — weight-only, activations stay fp16). Prototype: the
// decode / small-m path; prefill keeps the cuBLASLt path for now.
#pragma once
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// W: quantized weight, n_out rows each of n_in values in native GGUF blocks (row-major,
// blocks run along `in`). x: [m, n_in] fp16 (row-major). y: [m, n_out] fp16 (row-major).
// One CUDA block per (output row, token); threads reduce over n_in.
void fused_gemv_q8_0(const uint8_t* W, const __half* x, __half* y, int m, int n_out, int n_in, cudaStream_t stream);
void fused_gemv_q4_K(const uint8_t* W, const __half* x, __half* y, int m, int n_out, int n_in, cudaStream_t stream);

// Dispatch by ggml type; returns false if the type has no fused kernel yet (caller
// falls back to dequant + cuBLAS).
bool fused_gemv(const uint8_t* W, const __half* x, __half* y, int m, int n_out, int n_in,
                uint32_t ggml_type, cudaStream_t stream);

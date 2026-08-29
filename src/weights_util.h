// Shared host weight helpers (used by the dense loader and the MoE loader).
#pragma once
#include "loader.h"
#include <cuda_fp16.h>
#include <vector>

// Dequant n elements of a GGUF tensor into fp16 (host). F32/F16/Q8_0/Q4_K/Q6_K.
bool dequant_host(const uint8_t* data, uint32_t type, size_t n, __half* dst);

// Transpose src[rows, cols] (row-major) -> dst[cols, rows] (row-major), fp16.
void transpose_host(const __half* src, __half* dst, int rows, int cols);

// Re-quantize fp16 -> Q8_0 (34 B/32) or Q4_0 (18 B/32) for streaming.
void quantize_q8_0(const __half* x, size_t n, uint8_t* out);
void quantize_q4_0(const __half* x, size_t n, uint8_t* out);

// out = ne1 (rows), in = ne0 (cols) of a GGUF weight tensor.
void weight_out_in(const TensorInfo& t, int* out, int* in);

// Dequant a whole tensor into a temp fp16 buffer.
bool dequant_tensor(const GgufFile& g, const TensorInfo& t, std::vector<__half>& tmp);

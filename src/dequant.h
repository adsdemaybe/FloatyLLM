// Q8_0 dequantization: GGUF block format -> fp16.
#pragma once
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Elements per Q8_0 block.
static constexpr int QK8_0 = 32;

// GGUF block_q8_0: 32 int8 quants sharing one fp16 scale. 34 bytes, matches ggml.
struct __align__(2) BlockQ80 {
    __half d;         // scale
    int8_t qs[QK8_0]; // quantized values
};
static_assert(sizeof(BlockQ80) == 34, "BlockQ80 must match GGUF layout (34 bytes)");

// Dequantize n_blocks Q8_0 blocks -> out (n_blocks*32 fp16). x = d * q. Async on stream.
void dequant_q8_0(const BlockQ80* d_blocks, __half* d_out, int n_blocks, cudaStream_t stream);

// GGUF block_q4_0: 18 bytes / 32 values = fp16 scale + 16 packed nibble pairs.
// low nibble -> element j, high nibble -> element j+16. x = d*(q - 8).
struct __align__(2) BlockQ40 {
    __half d;
    uint8_t qs[16];
};
static_assert(sizeof(BlockQ40) == 18, "BlockQ40 must be 18 bytes");

void dequant_q4_0(const BlockQ40* d_blocks, __half* d_out, int n_blocks, cudaStream_t stream);

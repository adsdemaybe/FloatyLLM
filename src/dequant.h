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

// K-quant super-blocks (256 values). GPU dequant, so original Q4_K/Q6_K weights can
// be streamed and dequantized on the device (no host requant).
struct __align__(2) BlockQ3K { uint8_t hmask[32]; uint8_t qs[64]; uint8_t scales[12]; __half d; };  // 110 B
struct __align__(2) BlockQ4K { __half d, dmin; uint8_t scales[12]; uint8_t qs[128]; };  // 144 B
struct __align__(2) BlockQ5K { __half d, dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; };  // 176 B
struct __align__(2) BlockQ6K { uint8_t ql[128], qh[64]; int8_t scales[16]; __half d; };  // 210 B
static_assert(sizeof(BlockQ3K) == 110, "BlockQ3K must be 110 bytes");
static_assert(sizeof(BlockQ4K) == 144, "BlockQ4K must be 144 bytes");
static_assert(sizeof(BlockQ5K) == 176, "BlockQ5K must be 176 bytes");
static_assert(sizeof(BlockQ6K) == 210, "BlockQ6K must be 210 bytes");

void dequant_q3_K(const BlockQ3K* d_blocks, __half* d_out, int n_blocks, cudaStream_t stream);
void dequant_q4_K(const BlockQ4K* d_blocks, __half* d_out, int n_blocks, cudaStream_t stream);
void dequant_q5_K(const BlockQ5K* d_blocks, __half* d_out, int n_blocks, cudaStream_t stream);
void dequant_q6_K(const BlockQ6K* d_blocks, __half* d_out, int n_blocks, cudaStream_t stream);

// Transpose fp16 src[rows,cols] (row-major) -> dst[cols,rows] on the GPU.
void transpose_fp16(const __half* src, __half* dst, int rows, int cols, cudaStream_t stream);

// Dequant n elements of a tensor region (device quant/float bytes) -> fp16, by ggml
// type (F32/F16/Q4_0/Q8_0/Q4_K/Q6_K). Lets original quantized weights be streamed
// and dequantized on the GPU.
void dequant_to_fp16(const uint8_t* q, __half* out, uint32_t ggml_type, size_t n, cudaStream_t stream);

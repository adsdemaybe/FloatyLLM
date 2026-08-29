// K-quant dequantization (Q4_K, Q6_K) -> fp32. Host-only, ported from the ggml
// reference. Q4_K_M models use these (plus Q8_0/F32). PLAN sec 11. Output float
// so this stays CUDA-free and CI-testable; the caller converts to fp16.
#pragma once
#include <cstdint>
#include <cstddef>

// QK_K super-block size for K-quants.
static constexpr int QK_K = 256;

// Q4_K block = 144 bytes / 256 weights: d(fp16) dmin(fp16) scales[12] qs[128].
void dequant_q4_K_f32(const uint8_t* data, size_t n, float* dst);

// Q6_K block = 210 bytes / 256 weights: ql[128] qh[64] scales[16] d(fp16).
void dequant_q6_K_f32(const uint8_t* data, size_t n, float* dst);

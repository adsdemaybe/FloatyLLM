// Portable (no-CUDA) model view for the backend-agnostic runner. Parses a dense Llama GGUF
// via the host-only loader, exposes per-layer quantized weight pointers (read straight from
// the mmap by whichever Backend), and CPU-dequantizes the small always-fp16 pieces (norms,
// embedding table). Shared by the CUDA (GB10) and Metal (Apple) builds.
#pragma once
#include "loader.h"
#include <cstdint>
#include <string>
#include <vector>

// One weight matrix: where its quantized bytes live in the mmap + its shape/type.
struct FMat {
    const uint8_t* src = nullptr;
    uint32_t type = 0;
    int out = 0, in = 0;
    size_t quant_bytes = 0;
};

struct FLayer {
    FMat wq, wk, wv, wo, gate, up, down;   // quantized, read via Backend::wrap_host
    std::vector<uint16_t> attn_norm, ffn_norm;   // fp16 (dequantized on CPU at load)
};

struct FModel {
    GgufFile g;
    std::string arch;
    int n_layers = 0, dim = 0, n_heads = 0, n_kv_heads = 0, head_dim = 0, ffn = 0, vocab = 0;
    float eps = 1e-5f, rope_base = 10000.0f;
    std::vector<FLayer> layers;
    std::vector<uint16_t> token_embd;   // fp16 [vocab, dim]
    std::vector<uint16_t> final_norm;   // fp16 [dim]
    FMat output;                        // LM head weight (quantized) — via gemv
};

// Load a dense Llama GGUF. Returns false + *err on failure (incl. non-dense / unsupported).
bool floaty_load(const char* path, FModel* m, std::string* err);
void floaty_free(FModel* m);

// fp16 helpers (host).
uint16_t f32_to_f16(float f);
float    f16_to_f32(uint16_t h);

// CPU-dequantize n elements of a tensor region -> fp16. Supports F32/F16/Q4_0/Q8_0/Q2_K..Q6_K.
void cpu_dequant(const uint8_t* q, uint16_t* out, uint32_t ggml_type, size_t n);

// Bytes / elements for a ggml type block (be = elems per block, bb = bytes per block).
void ggml_block_info(uint32_t type, size_t* be, size_t* bb);

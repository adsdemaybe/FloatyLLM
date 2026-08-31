// Backend abstraction: the FloatyLLM streaming scheduler (layer loop, KV cache, budget /
// residency, sampling) is portable C++; only the compute kernels are device-specific. This
// interface is the seam. A CUDA backend (NVIDIA / GB10) wraps the existing .cu kernels; a
// Metal backend (Apple Silicon) wraps the MSL kernels in src/metal/. Both target UNIFIED
// memory -- weights are read straight from the mmap'd model (wrap_host), no PCIe copy.
//
// Buffers are opaque void* handles (a CUDA device pointer, or an id<MTLBuffer>). All compute
// buffers are fp16 unless noted; token-id / position buffers are int32.
#pragma once
#include <cstdint>
#include <cstddef>

struct Backend {
    virtual ~Backend() {}
    virtual const char* name() const = 0;

    // --- memory ---
    virtual void* alloc(size_t bytes) = 0;                                  // device-resident buffer
    virtual void  release(void* buf) = 0;
    virtual void  upload(void* dst, const void* host, size_t bytes) = 0;    // host -> device
    virtual void  download(void* host, const void* src, size_t bytes) = 0;  // device -> host
    // Zero-copy device-readable view of a host pointer (e.g. the mmap'd model). On unified
    // memory this aliases the same physical pages -- no copy. Returns a handle usable as a
    // buffer arg; release() frees the view (not the underlying host memory).
    virtual void* wrap_host(const void* host, size_t bytes) = 0;
    virtual void  sync() = 0;

    // --- kernels ---
    // y[m,n_out] = x[m,n_in] * dequant(W[n_out,n_in]) reading W in native GGUF quant layout.
    // y_off_elems shifts the output write (fp16 elems) -- e.g. to write K/V into a KV cache
    // at a token/layer offset.
    virtual void fused_gemv(const void* W, const void* x, void* y, int m, int n_out, int n_in,
                            uint32_t ggml_type, size_t y_off_elems = 0, size_t x_off_elems = 0) = 0;
    virtual void rmsnorm(const void* x, const void* w, void* out, int n_rows, int dim, float eps) = 0;
    virtual void rope(void* x, const void* positions, int n_tokens, int n_heads, int head_dim, float base,
                      size_t x_off_elems = 0) = 0;
    // Kcache/Vcache read from their layer base via k_off/v_off (fp16 elems).
    virtual void attention(const void* Q, const void* Kcache, size_t k_off, const void* Vcache, size_t v_off,
                           void* out, int n_new, int len_before, int n_heads, int n_kv_heads, int head_dim, float scale) = 0;
    virtual void silu_mul(void* gate, const void* up, int n) = 0;
    virtual void residual_add(void* x, const void* y, int n) = 0;
    virtual void embed(const void* table, const void* ids, void* out, int n, int dim) = 0;
    virtual int  argmax(const void* logits, int n) = 0;   // greedy sample: returns token id
};

// Factories (defined per backend TU; nullptr if that backend was not built in).
Backend* make_cuda_backend();
Backend* make_metal_backend();

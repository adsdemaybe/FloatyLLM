// CUDA backend: implements the Backend interface by wrapping the existing .cu kernels.
// On GB10 unified memory the GPU reads host mmap pointers directly (pageableMemoryAccess),
// so wrap_host is a zero-copy passthrough -- the same weights-in-unified-RAM path the Metal
// backend gets via newBufferWithBytesNoCopy.
#include "backend.h"
#include "fused_gemm.h"
#include "rmsnorm.h"
#include "rope.h"
#include "attention.h"
#include "elementwise.h"
#include "runner.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>

namespace {

__global__ void argmax_kernel(const __half* logits, int n, int* out) {
    __shared__ float sval[256]; __shared__ int sidx[256];
    int t = threadIdx.x; float best = -1e30f; int bi = 0;
    for (int v = t; v < n; v += blockDim.x) { float f = __half2float(logits[v]); if (f > best) { best = f; bi = v; } }
    sval[t] = best; sidx[t] = bi; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s && sval[t + s] > sval[t]) { sval[t] = sval[t + s]; sidx[t] = sidx[t + s]; }
        __syncthreads();
    }
    if (t == 0) out[0] = sidx[0];
}

class CudaBackend : public Backend {
public:
    cudaStream_t st = 0;
    int* d_arg = nullptr;
    CudaBackend() { cudaStreamCreate(&st); cudaMalloc(&d_arg, 4); }
    ~CudaBackend() override { if (d_arg) cudaFree(d_arg); if (st) cudaStreamDestroy(st); }

    const char* name() const override { return "cuda"; }

    void* alloc(size_t bytes) override { void* p = nullptr; cudaMalloc(&p, bytes); return p; }
    void  release(void* buf) override { if (buf) cudaFree(buf); }
    void  upload(void* dst, const void* host, size_t bytes) override { cudaMemcpyAsync(dst, host, bytes, cudaMemcpyHostToDevice, st); }
    void  download(void* host, const void* src, size_t bytes) override { cudaMemcpy(host, src, bytes, cudaMemcpyDeviceToHost); }
    // GB10: the kernels dereference host pointers directly -> no copy, just pass it through.
    void* wrap_host(const void* host, size_t) override { return const_cast<void*>(host); }
    void  sync() override { cudaStreamSynchronize(st); }

    void fused_gemv(const void* W, const void* x, void* y, int m, int n_out, int n_in, uint32_t type) override {
        ::fused_gemv((const uint8_t*)W, (const __half*)x, (__half*)y, m, n_out, n_in, type, st);
    }
    void rmsnorm(const void* x, const void* w, void* out, int n_rows, int dim, float eps) override {
        ::rmsnorm((const __half*)x, (const __half*)w, (__half*)out, n_rows, dim, eps, st);
    }
    void rope(void* x, const void* pos, int n_tokens, int n_heads, int head_dim, float base) override {
        ::rope_inplace((__half*)x, (const int*)pos, n_tokens, n_heads, head_dim, base, st);
    }
    void attention(const void* Q, const void* Kc, const void* Vc, void* out, int n_new, int len_before,
                   int n_heads, int n_kv_heads, int head_dim, float scale) override {
        ::attention_cached((const __half*)Q, (const __half*)Kc, (const __half*)Vc, (__half*)out,
                           n_new, len_before, n_heads, n_kv_heads, head_dim, scale, st);
    }
    void silu_mul(void* gate, const void* up, int n) override {
        ::silu((const __half*)gate, (__half*)gate, n, st);
        ::elementwise_mul((const __half*)gate, (const __half*)up, (__half*)gate, n, st);
    }
    void residual_add(void* x, const void* y, int n) override { ::residual_add((__half*)x, (const __half*)y, n, st); }
    void embed(const void* table, const void* ids, void* out, int n, int dim) override {
        ::embed_tokens((const __half*)table, (const int*)ids, (__half*)out, n, dim, st);
    }
    int argmax(const void* logits, int n) override {
        argmax_kernel<<<1, 256, 0, st>>>((const __half*)logits, n, d_arg);
        int idx = 0; cudaMemcpy(&idx, d_arg, 4, cudaMemcpyDeviceToHost);
        return idx;
    }
};

}  // namespace

Backend* make_cuda_backend() { return new CudaBackend(); }

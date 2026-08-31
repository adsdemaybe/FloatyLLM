// Metal backend: implements the Backend interface with the MSL kernels in floaty_kernels.metal.
// Apple Silicon unified memory -> MTLResourceStorageModeShared buffers alias host RAM, and
// wrap_host uses newBufferWithBytesNoCopy so the GPU reads the mmap'd model with no copy
// (the Metal analogue of GB10's pageableMemoryAccess).
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "../backend.h"
#include <cstdio>
#include <cstring>
#include <string>

namespace {

struct GemvDims { int m, n_out, n_in; };
struct NormArgs { int n_rows, dim; float eps; };
struct RopeArgs { int n_tokens, n_heads, head_dim; float base; };
struct AttnArgs { int n_new, len_before, n_heads, n_kv_heads, head_dim; float scale; };

// GGML type -> gemv kernel name.
static const char* gemv_name(uint32_t t) {
    switch (t) {
        case 10: return "gemv_q2_K"; case 11: return "gemv_q3_K"; case 12: return "gemv_q4_K";
        case 13: return "gemv_q5_K"; case 14: return "gemv_q6_K"; default: return nullptr;
    }
}

class MetalBackend : public Backend {
public:
    id<MTLDevice> dev = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> lib = nil;
    NSMutableDictionary* pipes = nil;   // name -> id<MTLComputePipelineState>
    NSMutableArray* live = nil;         // retain allocated/wrapped buffers
    id<MTLCommandBuffer> cmd = nil;     // current batch

    bool init(std::string* err) {
        dev = MTLCreateSystemDefaultDevice();
        if (!dev) { if (err) *err = "no Metal device"; return false; }
        queue = [dev newCommandQueue];
        pipes = [NSMutableDictionary dictionary];
        live = [NSMutableArray array];
        // Load floaty_kernels.metallib from beside the executable (or FLOATY_METALLIB).
        NSString* path = nil;
        if (const char* env = getenv("FLOATY_METALLIB")) path = [NSString stringWithUTF8String:env];
        if (!path) {
            NSString* exe = [[NSBundle mainBundle] executablePath];
            path = [[exe stringByDeletingLastPathComponent] stringByAppendingPathComponent:@"floaty_kernels.metallib"];
        }
        NSError* e = nil;
        lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:path] error:&e];
        if (!lib) { if (err) *err = std::string("load metallib: ") + [[e localizedDescription] UTF8String]; return false; }
        return true;
    }

    id<MTLComputePipelineState> pipe(const char* name) {
        NSString* key = [NSString stringWithUTF8String:name];
        id<MTLComputePipelineState> p = pipes[key];
        if (p) return p;
        id<MTLFunction> fn = [lib newFunctionWithName:key];
        NSError* e = nil;
        p = [dev newComputePipelineStateWithFunction:fn error:&e];
        if (p) pipes[key] = p;
        return p;
    }

    id<MTLComputeCommandEncoder> begin() {
        if (!cmd) cmd = [queue commandBuffer];
        return [cmd computeCommandEncoder];
    }

    const char* name() const override { return "metal"; }

    void* alloc(size_t bytes) override {
        id<MTLBuffer> b = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        [live addObject:b];
        return (__bridge void*)b;
    }
    void release(void* buf) override {
        if (!buf) return;
        id<MTLBuffer> b = (__bridge id<MTLBuffer>)buf;
        [live removeObject:b];
    }
    void upload(void* dst, const void* host, size_t bytes) override {
        id<MTLBuffer> b = (__bridge id<MTLBuffer>)dst; memcpy([b contents], host, bytes);
    }
    void download(void* host, const void* src, size_t bytes) override {
        id<MTLBuffer> b = (__bridge id<MTLBuffer>)src; memcpy(host, [b contents], bytes);
    }
    void* wrap_host(const void* host, size_t bytes) override {
        // newBufferWithBytesNoCopy needs a page-aligned pointer + length. The mmap'd model is
        // page-aligned; round the length up to a page.
        size_t pg = 16384;  // Apple Silicon page size
        size_t len = (bytes + pg - 1) & ~(pg - 1);
        id<MTLBuffer> b = [dev newBufferWithBytesNoCopy:(void*)host length:len
                                                options:MTLResourceStorageModeShared deallocator:nil];
        if (!b) return nullptr;
        [live addObject:b];
        return (__bridge void*)b;
    }
    void sync() override {
        if (!cmd) return;
        [cmd commit]; [cmd waitUntilCompleted]; cmd = nil;
    }

    void setDims(id<MTLComputeCommandEncoder> enc, int idx, const void* p, size_t n) {
        [enc setBytes:p length:n atIndex:idx];
    }

    void fused_gemv(const void* W, const void* x, void* y, int m, int n_out, int n_in, uint32_t type,
                    size_t y_off = 0, size_t x_off = 0) override {
        const char* kn = gemv_name(type); if (!kn) return;
        id<MTLComputeCommandEncoder> enc = begin();
        [enc setComputePipelineState:pipe(kn)];
        [enc setBuffer:(__bridge id<MTLBuffer>)W offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)x offset:x_off * 2 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)y offset:y_off * 2 atIndex:2];
        GemvDims d{m, n_out, n_in}; setDims(enc, 3, &d, sizeof(d));
        [enc dispatchThreadgroups:MTLSizeMake((n_out + 7) / 8, m, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    void rmsnorm(const void* x, const void* w, void* out, int n_rows, int dim, float eps) override {
        id<MTLComputeCommandEncoder> enc = begin();
        [enc setComputePipelineState:pipe("rmsnorm")];
        [enc setBuffer:(__bridge id<MTLBuffer>)x offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)w offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)out offset:0 atIndex:2];
        NormArgs a{n_rows, dim, eps}; setDims(enc, 3, &a, sizeof(a));
        [enc dispatchThreadgroups:MTLSizeMake(n_rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [enc endEncoding];
    }
    void rope(void* x, const void* pos, int n_tokens, int n_heads, int head_dim, float base,
              size_t x_off = 0) override {
        id<MTLComputeCommandEncoder> enc = begin();
        [enc setComputePipelineState:pipe("rope")];
        [enc setBuffer:(__bridge id<MTLBuffer>)x offset:x_off * 2 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)pos offset:0 atIndex:1];
        RopeArgs a{n_tokens, n_heads, head_dim, base}; setDims(enc, 2, &a, sizeof(a));
        int total = n_tokens * n_heads * (head_dim / 2);
        [enc dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    void attention(const void* Q, const void* Kc, size_t k_off, const void* Vc, size_t v_off, void* out,
                   int n_new, int len_before, int n_heads, int n_kv_heads, int head_dim, float scale) override {
        id<MTLComputeCommandEncoder> enc = begin();
        [enc setComputePipelineState:pipe("attention")];
        [enc setBuffer:(__bridge id<MTLBuffer>)Q offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)Kc offset:k_off * 2 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)Vc offset:v_off * 2 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)out offset:0 atIndex:3];
        AttnArgs a{n_new, len_before, n_heads, n_kv_heads, head_dim, scale}; setDims(enc, 4, &a, sizeof(a));
        [enc dispatchThreadgroups:MTLSizeMake(n_new, n_heads, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [enc endEncoding];
    }
    void silu_mul(void* gate, const void* up, int n) override { elem2("silu_mul", gate, up, n); }
    void residual_add(void* x, const void* y, int n) override { elem2("residual_add", x, y, n); }
    void elem2(const char* kn, void* a0, const void* a1, int n) {
        id<MTLComputeCommandEncoder> enc = begin();
        [enc setComputePipelineState:pipe(kn)];
        [enc setBuffer:(__bridge id<MTLBuffer>)a0 offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)a1 offset:0 atIndex:1];
        setDims(enc, 2, &n, sizeof(n));
        [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    void embed(const void* table, const void* ids, void* out, int n, int dim) override {
        id<MTLComputeCommandEncoder> enc = begin();
        [enc setComputePipelineState:pipe("embed")];
        [enc setBuffer:(__bridge id<MTLBuffer>)table offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)ids offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)out offset:0 atIndex:2];
        int nd[2] = {n, dim}; setDims(enc, 3, nd, sizeof(nd));
        [enc dispatchThreads:MTLSizeMake(n * dim, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    int argmax(const void* logits, int n) override {
        id<MTLBuffer> res = [dev newBufferWithLength:sizeof(int) options:MTLResourceStorageModeShared];
        id<MTLComputeCommandEncoder> enc = begin();
        [enc setComputePipelineState:pipe("argmax")];
        [enc setBuffer:(__bridge id<MTLBuffer>)logits offset:0 atIndex:0];
        setDims(enc, 1, &n, sizeof(n));
        [enc setBuffer:res offset:0 atIndex:2];
        [enc setThreadgroupMemoryLength:8 * sizeof(float) atIndex:0];
        [enc setThreadgroupMemoryLength:8 * sizeof(int) atIndex:1];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        sync();
        return *(const int*)[res contents];
    }
};

}  // namespace

Backend* make_metal_backend() {
    MetalBackend* b = new MetalBackend();
    std::string err;
    if (!b->init(&err)) { fprintf(stderr, "metal backend init failed: %s\n", err.c_str()); delete b; return nullptr; }
    return b;
}

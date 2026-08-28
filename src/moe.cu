// MoE MLP implementation.
#include "moe.h"

// One thread per token: softmax over experts, keep top-k, renormalize.
__global__ void router_topk_kernel(const __half* logits, float* route_w,
                                   int T, int E, int topk) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    const __half* lr = logits + (size_t)t * E;
    float* w = route_w + (size_t)t * E;

    float m = -1e30f;
    for (int e = 0; e < E; ++e) m = fmaxf(m, __half2float(lr[e]));
    float sum = 0.0f;
    for (int e = 0; e < E; ++e) { float p = expf(__half2float(lr[e]) - m); w[e] = p; sum += p; }
    for (int e = 0; e < E; ++e) w[e] /= sum;   // softmax probs

    // Zero all but the top-k, tracking their sum for renormalization.
    float topsum = 0.0f;
    for (int pick = 0; pick < topk; ++pick) {
        float best = -1.0f; int bi = -1;
        for (int e = 0; e < E; ++e) if (w[e] >= 0.0f && w[e] > best) { best = w[e]; bi = e; }
        if (bi < 0) break;
        topsum += w[bi];
        w[bi] = -w[bi] - 1.0f;   // mark selected (negative encoding, < -1)
    }
    for (int e = 0; e < E; ++e) {
        if (w[e] < -0.5f) w[e] = (-w[e] - 1.0f) / topsum;   // restore + renormalize
        else w[e] = 0.0f;                                   // not selected
    }
}

// out[t,:] += route_w[t,e] * ye[t,:]
__global__ void scale_accumulate_kernel(__half* out, const __half* ye,
                                        const float* route_w, int e, int E,
                                        int T, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * dim) return;
    int t = i / dim;
    float w = route_w[(size_t)t * E + e];
    if (w == 0.0f) return;
    out[i] = __float2half(__half2float(out[i]) + w * __half2float(ye[i]));
}

__global__ void zero_kernel(__half* p, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = __float2half(0.0f);
}

__global__ void silu_mul_kernel(__half* gate, const __half* up, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = __half2float(gate[i]);
    gate[i] = __float2half((g / (1.0f + expf(-g))) * __half2float(up[i]));
}

void moe_router(Gemm* g, const __half* x, const __half* w_gate, __half* logits_TE,
                float* route_w, int T, int dim, int n_experts, int topk,
                cudaStream_t stream) {
    gemm_rowmajor(g, x, w_gate, logits_TE, T, n_experts, dim, stream);
    int t = 128;
    router_topk_kernel<<<(T + t - 1) / t, t, 0, stream>>>(logits_TE, route_w, T, n_experts, topk);
}

void moe_mlp(Gemm* g, const __half* x, const __half* const* wgate,
             const __half* const* wup, const __half* const* wdown,
             const float* route_w, __half* out, __half* gate, __half* up, __half* ye,
             int T, int dim, int ffn, int n_experts, cudaStream_t stream) {
    int td = T * dim, tf = T * ffn, t = 256;
    zero_kernel<<<(td + t - 1) / t, t, 0, stream>>>(out, td);
    for (int e = 0; e < n_experts; ++e) {
        gemm_rowmajor(g, x, wgate[e], gate, T, ffn, dim, stream);
        gemm_rowmajor(g, x, wup[e], up, T, ffn, dim, stream);
        silu_mul_kernel<<<(tf + t - 1) / t, t, 0, stream>>>(gate, up, tf);
        gemm_rowmajor(g, gate, wdown[e], ye, T, dim, ffn, stream);
        scale_accumulate_kernel<<<(td + t - 1) / t, t, 0, stream>>>(out, ye, route_w, e, n_experts, T, dim);
    }
}

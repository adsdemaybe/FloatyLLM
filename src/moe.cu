// MoE MLP implementation.
#include "moe.h"
#include "rmsnorm.h"
#include "rope.h"
#include "attention.h"
#include "elementwise.h"

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

void moe_mlp_zero(__half* out, int n, cudaStream_t stream) {
    int t = 256;
    zero_kernel<<<(n + t - 1) / t, t, 0, stream>>>(out, n);
}

// One expert's SwiGLU: out[t,:] += route_w[t,e] * down_e(silu(gate_e(x)) * up_e(x)).
// Reads wgate_e/wup_e/wdown_e (that expert's slot); does NOT zero out (caller does once).
void moe_mlp_one(Gemm* g, const __half* x, const __half* wgate_e, const __half* wup_e,
                 const __half* wdown_e, const float* route_w, __half* out, __half* gate,
                 __half* up, __half* ye, int T, int dim, int ffn, int e, int n_experts,
                 cudaStream_t stream) {
    int td = T * dim, tf = T * ffn, t = 256;
    gemm_rowmajor(g, x, wgate_e, gate, T, ffn, dim, stream);
    gemm_rowmajor(g, x, wup_e, up, T, ffn, dim, stream);
    silu_mul_kernel<<<(tf + t - 1) / t, t, 0, stream>>>(gate, up, tf);
    gemm_rowmajor(g, gate, wdown_e, ye, T, dim, ffn, stream);
    scale_accumulate_kernel<<<(td + t - 1) / t, t, 0, stream>>>(out, ye, route_w, e, n_experts, T, dim);
}

void moe_mlp(Gemm* g, const __half* x, const __half* const* wgate,
             const __half* const* wup, const __half* const* wdown,
             const float* route_w, __half* out, __half* gate, __half* up, __half* ye,
             int T, int dim, int ffn, int n_experts, const int* active, int n_active,
             cudaStream_t stream) {
    moe_mlp_zero(out, T * dim, stream);
    for (int a = 0; a < n_active; ++a) {
        int e = active[a];
        moe_mlp_one(g, x, wgate[e], wup[e], wdown[e], route_w, out, gate, up, ye,
                    T, dim, ffn, e, n_experts, stream);
    }
}

int moe_attn_route(const LlamaConfig& cfg, const MoeConfig& mcfg, const MoeLayerWeights& w,
                   __half* hidden, const int* d_positions, int n_new, int layer_idx, int len_before,
                   KVCache& kv, LayerScratch& s, MoeScratch& ms, Gemm* gemm, cudaStream_t stream,
                   int* active, int max_active) {
    int dim = cfg.dim, qd = cfg.n_heads * cfg.head_dim, kvd = cfg.n_kv_heads * cfg.head_dim;
    float scale = 1.0f / sqrtf((float)cfg.head_dim);

    // --- attention block (same as layer_forward_cached) ---
    __half* Kbase = kv.K + (size_t)layer_idx * kv.max_T * kvd;
    __half* Vbase = kv.V + (size_t)layer_idx * kv.max_T * kvd;
    __half* Kdst = Kbase + (size_t)len_before * kvd;
    __half* Vdst = Vbase + (size_t)len_before * kvd;
    rmsnorm(hidden, w.attn_norm, s.xn, n_new, dim, cfg.eps, stream);
    gemm_rowmajor(gemm, s.xn, w.wq, s.q, n_new, qd, dim, stream);
    gemm_rowmajor(gemm, s.xn, w.wk, Kdst, n_new, kvd, dim, stream);
    gemm_rowmajor(gemm, s.xn, w.wv, Vdst, n_new, kvd, dim, stream);
    rope_inplace(s.q, d_positions, n_new, cfg.n_heads, cfg.head_dim, cfg.rope_base, stream);
    rope_inplace(Kdst, d_positions, n_new, cfg.n_kv_heads, cfg.head_dim, cfg.rope_base, stream);
    attention_cached(s.q, Kbase, Vbase, s.att, n_new, len_before,
                     cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, scale, stream);
    gemm_rowmajor(gemm, s.att, w.wo, s.proj, n_new, dim, qd, stream);
    residual_add(hidden, s.proj, n_new * dim, stream);

    // --- router: ffn_norm(-> s.xn) then top-k gate. s.xn is the expert-block input,
    // preserved for the caller's moe_experts_out (survives expert streaming). ---
    rmsnorm(hidden, w.ffn_norm, s.xn, n_new, dim, cfg.eps, stream);
    moe_router(gemm, s.xn, w.w_gate, ms.logits, ms.route_w, n_new, dim,
               mcfg.n_experts, mcfg.n_used, stream);
    // Pick the active experts (any token routed to them). For decode (n_new=1) this is
    // exactly n_used experts -> the caller streams only those, not all n_experts.
    cudaMemcpyAsync(ms.h_route, ms.route_w, (size_t)n_new * mcfg.n_experts * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    int n_active = 0;
    for (int e = 0; e < mcfg.n_experts && n_active < max_active; ++e) {
        bool used = false;
        for (int tk = 0; tk < n_new; ++tk) if (ms.h_route[(size_t)tk * mcfg.n_experts + e] != 0.0f) { used = true; break; }
        if (used) active[n_active++] = e;
    }
    return n_active;
}

// Shared expert (if any) added to ms.moe_out, then hidden += moe_out. Call after the
// routed experts have accumulated into ms.moe_out.
void moe_experts_tail(const LlamaConfig& cfg, const MoeConfig& mcfg, const MoeLayerWeights& w,
                      __half* hidden, int n_new, LayerScratch& s, MoeScratch& ms, Gemm* gemm,
                      cudaStream_t stream) {
    int dim = cfg.dim;
    if (mcfg.has_shared && w.sh_gate) {
        gemm_rowmajor(gemm, s.xn, w.sh_gate, ms.gate, n_new, mcfg.expert_ffn, dim, stream);
        gemm_rowmajor(gemm, s.xn, w.sh_up, ms.up, n_new, mcfg.expert_ffn, dim, stream);
        silu(ms.gate, ms.gate, n_new * mcfg.expert_ffn, stream);
        elementwise_mul(ms.gate, ms.up, ms.gate, n_new * mcfg.expert_ffn, stream);
        gemm_rowmajor(gemm, ms.gate, w.sh_down, ms.ye, n_new, dim, mcfg.expert_ffn, stream);
        residual_add(ms.moe_out, ms.ye, n_new * dim, stream);
    }
    residual_add(hidden, ms.moe_out, n_new * dim, stream);
}

void moe_experts_out(const LlamaConfig& cfg, const MoeConfig& mcfg, const MoeLayerWeights& w,
                     __half* hidden, int n_new, LayerScratch& s, MoeScratch& ms, Gemm* gemm,
                     cudaStream_t stream, const int* active, int n_active) {
    moe_mlp(gemm, s.xn, w.wgate, w.wup, w.wdown, ms.route_w, ms.moe_out,
            ms.gate, ms.up, ms.ye, n_new, cfg.dim, mcfg.expert_ffn, mcfg.n_experts, active, n_active, stream);
    moe_experts_tail(cfg, mcfg, w, hidden, n_new, s, ms, gemm, stream);
}

// Convenience wrapper (all experts resident): route then run. Streaming callers use
// moe_attn_route + moe_experts_out directly so they can stream only the active experts.
void moe_layer_forward_cached(const LlamaConfig& cfg, const MoeConfig& mcfg,
                              const MoeLayerWeights& w, __half* hidden, const int* d_positions,
                              int n_new, int layer_idx, int len_before, KVCache& kv,
                              LayerScratch& s, MoeScratch& ms, Gemm* gemm, cudaStream_t stream) {
    int active[256];
    int n_active = moe_attn_route(cfg, mcfg, w, hidden, d_positions, n_new, layer_idx, len_before,
                                  kv, s, ms, gemm, stream, active, 256);
    moe_experts_out(cfg, mcfg, w, hidden, n_new, s, ms, gemm, stream, active, n_active);
}

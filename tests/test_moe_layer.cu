// Unit test for the full MoE decoder layer assembly. RUN on a GPU.
// Checks finite output, and that zero output-projections (wo + expert wdown = 0)
// leave hidden unchanged (residual identity), confirming the wiring.
#include "moe.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

static __half* dev(const std::vector<float>& f) {
    std::vector<__half> h(f.size());
    for (size_t i = 0; i < f.size(); ++i) h[i] = __float2half(f[i]);
    __half* d; cudaMalloc(&d, h.size() * sizeof(__half));
    cudaMemcpy(d, h.data(), h.size() * sizeof(__half), cudaMemcpyHostToDevice);
    return d;
}

int main() {
    LlamaConfig cfg{32, 4, 2, 8, 0, 1e-5f, 10000.0f};   // ffn_dim unused for MoE
    MoeConfig mcfg; mcfg.n_experts = 4; mcfg.n_used = 2; mcfg.expert_ffn = 48; mcfg.has_shared = false;
    int dim = cfg.dim, qd = cfg.n_heads*cfg.head_dim, kvd = cfg.n_kv_heads*cfg.head_dim, ef = mcfg.expert_ffn;
    int n_new = 3;

    auto fill = [](std::vector<float>& x, float s, int seed) {
        for (size_t i = 0; i < x.size(); ++i) x[i] = s * (float)((int)((i*7+seed)%23) - 11); };

    std::vector<float> an(dim,1.0f), fn(dim,1.0f), wqf(dim*qd), wkf(dim*kvd), wvf(dim*kvd),
        wof(qd*dim), wgatef(dim*mcfg.n_experts), h0(n_new*dim);
    fill(wqf,0.05f,1); fill(wkf,0.05f,2); fill(wvf,0.05f,3); fill(wof,0.05f,4);
    fill(wgatef,0.1f,5); fill(h0,0.05f,9);

    std::vector<__half*> eg(mcfg.n_experts), eu(mcfg.n_experts), ed(mcfg.n_experts);
    std::vector<float> egf(dim*ef), euf(dim*ef), edf(ef*dim);
    fill(egf,0.05f,6); fill(euf,0.05f,7); fill(edf,0.05f,8);
    for (int e = 0; e < mcfg.n_experts; ++e) { eg[e]=dev(egf); eu[e]=dev(euf); ed[e]=dev(edf); }
    std::vector<const __half*> egc(eg.begin(),eg.end()), euc(eu.begin(),eu.end()), edc(ed.begin(),ed.end());

    MoeLayerWeights w{ dev(an), dev(wqf), dev(wkf), dev(wvf), dev(wof), dev(fn),
                       dev(wgatef), egc.data(), euc.data(), edc.data() };

    std::vector<int> pos = {0,1,2};
    int* dpos; cudaMalloc(&dpos, n_new*sizeof(int));
    cudaMemcpy(dpos, pos.data(), n_new*sizeof(int), cudaMemcpyHostToDevice);

    LayerScratch s;
    cudaMalloc(&s.xn,n_new*dim*2); cudaMalloc(&s.q,n_new*qd*2); cudaMalloc(&s.k,n_new*kvd*2);
    cudaMalloc(&s.v,n_new*kvd*2); cudaMalloc(&s.att,n_new*qd*2); cudaMalloc(&s.proj,n_new*dim*2);
    cudaMalloc(&s.gate,n_new*ef*2); cudaMalloc(&s.up,n_new*ef*2);
    MoeScratch ms;
    cudaMalloc(&ms.logits,n_new*mcfg.n_experts*2); cudaMalloc(&ms.route_w,n_new*mcfg.n_experts*sizeof(float));
    cudaMalloc(&ms.gate,n_new*ef*2); cudaMalloc(&ms.up,n_new*ef*2);
    cudaMalloc(&ms.ye,n_new*dim*2); cudaMalloc(&ms.moe_out,n_new*dim*2);
    cudaHostAlloc((void**)&ms.h_route, n_new*mcfg.n_experts*sizeof(float), cudaHostAllocDefault);

    KVCache kv; kv.n_layers=1; kv.max_T=n_new; kv.kvd=kvd; kv.len=0;
    cudaMalloc(&kv.K,(size_t)n_new*kvd*2); cudaMalloc(&kv.V,(size_t)n_new*kvd*2);

    Gemm g; gemm_create(&g);

    // Case 1: general run finite.
    __half* h1 = dev(h0);
    moe_layer_forward_cached(cfg, mcfg, w, h1, dpos, n_new, 0, 0, kv, s, ms, &g, 0);
    cudaDeviceSynchronize();
    std::vector<__half> out(n_new*dim);
    cudaMemcpy(out.data(), h1, out.size()*2, cudaMemcpyDeviceToHost);
    bool finite = true; for (auto v : out) if (!std::isfinite(__half2float(v))) finite = false;

    // Case 2: zero wo + zero expert wdown -> residual identity.
    std::vector<float> zqd(qd*dim,0.0f), zed(ef*dim,0.0f);
    MoeLayerWeights w2 = w; w2.wo = dev(zqd);
    std::vector<__half*> zd(mcfg.n_experts); std::vector<const __half*> zdc(mcfg.n_experts);
    for (int e=0;e<mcfg.n_experts;++e){ zd[e]=dev(zed); zdc[e]=zd[e]; }
    w2.wdown = zdc.data();
    kv.len = 0;
    __half* h2 = dev(h0);
    moe_layer_forward_cached(cfg, mcfg, w2, h2, dpos, n_new, 0, 0, kv, s, ms, &g, 0);
    cudaDeviceSynchronize();
    std::vector<__half> out2(n_new*dim);
    cudaMemcpy(out2.data(), h2, out2.size()*2, cudaMemcpyDeviceToHost);
    bool ident = true;
    for (int i=0;i<n_new*dim;++i) if (fabsf(__half2float(out2[i]) - h0[i]) > 1e-2f) ident = false;

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 2; }
    gemm_destroy(&g);
    printf("moe layer: finite=%d residual_identity=%d\n", (int)finite, (int)ident);
    if (!finite || !ident) { printf("FAIL\n"); return 1; }
    printf("PASS: moe_layer_forward_cached (E=%d, top-%d)\n", mcfg.n_experts, mcfg.n_used);
    return 0;
}

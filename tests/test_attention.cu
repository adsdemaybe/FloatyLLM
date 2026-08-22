// Unit test for naive attention. RUN on a GPU. Compares GPU online-softmax
// attention vs a direct fp32 host reference (GQA + causal).
#include "attention.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

static void attention_ref(const float* Q, const float* K, const float* V, float* out,
                          int T, int H, int KVH, int D, float scale) {
    int group = H / KVH;
    std::vector<float> sc(T);
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < H; ++h) {
            int kvh = h / group;
            const float* q = Q + ((size_t)t * H + h) * D;
            float m = -1e30f;
            for (int j = 0; j <= t; ++j) {
                const float* k = K + ((size_t)j * KVH + kvh) * D;
                float s = 0.0f; for (int d = 0; d < D; ++d) s += q[d] * k[d];
                sc[j] = s * scale; m = fmaxf(m, sc[j]);
            }
            float den = 0.0f; for (int j = 0; j <= t; ++j) { sc[j] = expf(sc[j] - m); den += sc[j]; }
            float* o = out + ((size_t)t * H + h) * D;
            for (int d = 0; d < D; ++d) o[d] = 0.0f;
            for (int j = 0; j <= t; ++j) {
                const float* v = V + ((size_t)j * KVH + kvh) * D;
                float p = sc[j] / den;
                for (int d = 0; d < D; ++d) o[d] += p * v[d];
            }
        }
}

int main() {
    const int T = 6, H = 4, KVH = 2, D = 16; // GQA: 2 query heads per kv head
    const float scale = 1.0f / sqrtf((float)D);

    std::vector<__half> Q(T*H*D), K(T*KVH*D), V(T*KVH*D), out(T*H*D);
    std::vector<float> Qf(T*H*D), Kf(T*KVH*D), Vf(T*KVH*D), expected(T*H*D);
    for (size_t i = 0; i < Q.size(); ++i) { float v = 0.1f*(float)((i*5)%29-14); Q[i]=__float2half(v); Qf[i]=__half2float(Q[i]); }
    for (size_t i = 0; i < K.size(); ++i) { float v = 0.1f*(float)((i*3)%23-11); K[i]=__float2half(v); Kf[i]=__half2float(K[i]); }
    for (size_t i = 0; i < V.size(); ++i) { float v = 0.1f*(float)((i*7)%31-15); V[i]=__float2half(v); Vf[i]=__half2float(V[i]); }

    attention_ref(Qf.data(), Kf.data(), Vf.data(), expected.data(), T, H, KVH, D, scale);

    __half *dQ, *dK, *dV, *dO;
    cudaMalloc(&dQ, Q.size()*sizeof(__half)); cudaMalloc(&dK, K.size()*sizeof(__half));
    cudaMalloc(&dV, V.size()*sizeof(__half)); cudaMalloc(&dO, out.size()*sizeof(__half));
    cudaMemcpy(dQ, Q.data(), Q.size()*sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(dK, K.data(), K.size()*sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(dV, V.data(), V.size()*sizeof(__half), cudaMemcpyHostToDevice);
    attention(dQ, dK, dV, dO, T, H, KVH, D, scale, 0);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 2; }
    cudaMemcpy(out.data(), dO, out.size()*sizeof(__half), cudaMemcpyDeviceToHost);
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);

    int fails = 0;
    for (size_t i = 0; i < out.size(); ++i) {
        float got = __half2float(out[i]);
        float tol = 2e-2f * (fabsf(expected[i]) + 1.0f);
        if (fabsf(got - expected[i]) > tol) { if (fails < 10) printf("mismatch i=%zu got=%f exp=%f\n", i, got, expected[i]); ++fails; }
    }
    if (fails) { printf("FAIL: %d/%zu\n", fails, out.size()); return 1; }
    printf("PASS: attention T=%d H=%d KVH=%d D=%d\n", T, H, KVH, D);
    return 0;
}

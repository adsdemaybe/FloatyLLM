// Naive causal attention kernel: one thread per (query token, head).
// Online softmax: running max m, running denom l, running output acc.
#include "attention.h"

static constexpr int MAX_HEAD_DIM = 128;

__global__ void attention_kernel(const __half* Q, const __half* K, const __half* V,
                                 __half* out, int n_tokens, int n_heads,
                                 int n_kv_heads, int head_dim, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_tokens * n_heads) return;
    int token = idx / n_heads;
    int head = idx % n_heads;
    int group = n_heads / n_kv_heads;
    int kvh = head / group;

    const __half* q = Q + ((size_t)token * n_heads + head) * head_dim;

    float acc[MAX_HEAD_DIM];
    for (int d = 0; d < head_dim; ++d) acc[d] = 0.0f;
    float m = -1e30f;
    float l = 0.0f;

    for (int j = 0; j <= token; ++j) {
        const __half* k = K + ((size_t)j * n_kv_heads + kvh) * head_dim;
        const __half* v = V + ((size_t)j * n_kv_heads + kvh) * head_dim;
        float s = 0.0f;
        for (int d = 0; d < head_dim; ++d) s += __half2float(q[d]) * __half2float(k[d]);
        s *= scale;

        float m_new = fmaxf(m, s);
        float corr = expf(m - m_new);
        float p = expf(s - m_new);
        l = l * corr + p;
        for (int d = 0; d < head_dim; ++d) acc[d] = acc[d] * corr + p * __half2float(v[d]);
        m = m_new;
    }

    __half* o = out + ((size_t)token * n_heads + head) * head_dim;
    float inv = 1.0f / l;
    for (int d = 0; d < head_dim; ++d) o[d] = __float2half(acc[d] * inv);
}

void attention(const __half* Q, const __half* K, const __half* V, __half* out,
               int n_tokens, int n_heads, int n_kv_heads, int head_dim,
               float scale, cudaStream_t stream) {
    if (n_tokens <= 0 || n_heads <= 0 || head_dim <= 0) return;
    int total = n_tokens * n_heads;
    int threads = 128;
    int blocks = (total + threads - 1) / threads;
    attention_kernel<<<blocks, threads, 0, stream>>>(
        Q, K, V, out, n_tokens, n_heads, n_kv_heads, head_dim, scale);
}

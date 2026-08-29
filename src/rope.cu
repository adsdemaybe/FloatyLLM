// RoPE kernel. One CUDA block per (token, head); each thread rotates one pair.
// angle_j = pos * base^(-2j/head_dim); rotate (x_2j, x_2j+1) by angle_j.
#include "rope.h"

__global__ void rope_kernel(__half* x, const int* positions, int n_heads,
                            int head_dim, float base) {
    int blk = blockIdx.x;            // token * n_heads + head
    int token = blk / n_heads;
    int pos = positions[token];
    __half* vec = x + (size_t)blk * head_dim;

    int half = head_dim / 2;
    for (int j = threadIdx.x; j < half; j += blockDim.x) {
        float freq = powf(base, -2.0f * (float)j / (float)head_dim);
        float angle = (float)pos * freq;
        float c = cosf(angle);
        float s = sinf(angle);
        float a = __half2float(vec[2 * j]);
        float b = __half2float(vec[2 * j + 1]);
        vec[2 * j]     = __float2half(a * c - b * s);
        vec[2 * j + 1] = __float2half(a * s + b * c);
    }
}

void rope_inplace(__half* x, const int* positions, int n_tokens, int n_heads,
                  int head_dim, float base, cudaStream_t stream) {
    if (n_tokens <= 0 || n_heads <= 0 || head_dim <= 0) return;
    int blocks = n_tokens * n_heads;
    int threads = 128;
    rope_kernel<<<blocks, threads, 0, stream>>>(x, positions, n_heads, head_dim, base);
}

// Elementwise kernels: SiLU, residual add, multiply. Grid-stride, fp32 math.
#include "elementwise.h"

__global__ void silu_kernel(const __half* x, __half* out, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        float v = __half2float(x[i]);
        out[i] = __float2half(v / (1.0f + expf(-v)));
    }
}

__global__ void residual_add_kernel(__half* x, const __half* y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        x[i] = __float2half(__half2float(x[i]) + __half2float(y[i]));
    }
}

__global__ void mul_kernel(const __half* a, const __half* b, __half* out, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        out[i] = __float2half(__half2float(a[i]) * __half2float(b[i]));
    }
}

static int grid_for(int n, int threads) {
    int blocks = (n + threads - 1) / threads;
    return blocks > 65535 ? 65535 : blocks;
}

void silu(const __half* x, __half* out, int n, cudaStream_t stream) {
    if (n <= 0) return;
    int t = 256;
    silu_kernel<<<grid_for(n, t), t, 0, stream>>>(x, out, n);
}

void residual_add(__half* x, const __half* y, int n, cudaStream_t stream) {
    if (n <= 0) return;
    int t = 256;
    residual_add_kernel<<<grid_for(n, t), t, 0, stream>>>(x, y, n);
}

void elementwise_mul(const __half* a, const __half* b, __half* out, int n,
                     cudaStream_t stream) {
    if (n <= 0) return;
    int t = 256;
    mul_kernel<<<grid_for(n, t), t, 0, stream>>>(a, b, out, n);
}

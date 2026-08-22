// Softmax kernel: one CUDA block per row. Max + sum reduced with CUB in fp32.
// out = exp(x - max) / sum(exp(x - max)).
#include "softmax.h"
#include <cub/block/block_reduce.cuh>

static constexpr int SOFTMAX_TPB = 256;

__global__ void softmax_kernel(const __half* x, __half* out, int dim) {
    int row = blockIdx.x;
    const __half* xr = x + (size_t)row * dim;
    __half* outr = out + (size_t)row * dim;

    typedef cub::BlockReduce<float, SOFTMAX_TPB> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float shared_max;
    __shared__ float shared_sum;

    float m = -1e30f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        m = fmaxf(m, __half2float(xr[i]));
    }
    float bm = BlockReduce(temp).Reduce(m, cub::Max());
    if (threadIdx.x == 0) shared_max = bm;
    __syncthreads();
    float mx = shared_max;

    float s = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        s += expf(__half2float(xr[i]) - mx);
    }
    __syncthreads();
    float bs = BlockReduce(temp).Sum(s);
    if (threadIdx.x == 0) shared_sum = bs;
    __syncthreads();
    float inv = 1.0f / shared_sum;

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        outr[i] = __float2half(expf(__half2float(xr[i]) - mx) * inv);
    }
}

void softmax(const __half* x, __half* out, int n_rows, int dim, cudaStream_t stream) {
    if (n_rows <= 0 || dim <= 0) return;
    softmax_kernel<<<n_rows, SOFTMAX_TPB, 0, stream>>>(x, out, dim);
}

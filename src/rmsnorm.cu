// RMSNorm kernel: one CUDA block per row. Sum-of-squares reduced with CUB
// (cub::BlockReduce, ships with the CUDA toolkit) in fp32 for stability.
// out_i = x_i * rsqrt(mean(x^2) + eps) * weight_i.  See PLAN.md section 7.
#include "rmsnorm.h"
#include <cub/block/block_reduce.cuh>

static constexpr int RMSNORM_TPB = 256;

__global__ void rmsnorm_kernel(const __half* x, const __half* w, __half* out,
                               int dim, float eps) {
    int row = blockIdx.x;
    const __half* xr = x + (size_t)row * dim;
    __half* outr = out + (size_t)row * dim;

    typedef cub::BlockReduce<float, RMSNORM_TPB> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float inv_rms;

    float local = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = __half2float(xr[i]);
        local += v * v;
    }

    float ss = BlockReduce(temp).Sum(local);
    if (threadIdx.x == 0) inv_rms = rsqrtf(ss / (float)dim + eps);
    __syncthreads();

    float inv = inv_rms;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = __half2float(xr[i]) * inv * __half2float(w[i]);
        outr[i] = __float2half(v);
    }
}

void rmsnorm(const __half* x, const __half* weight, __half* out,
             int n_rows, int dim, float eps, cudaStream_t stream) {
    if (n_rows <= 0 || dim <= 0) return;
    size_t shmem = 0;
    rmsnorm_kernel<<<n_rows, RMSNORM_TPB, shmem, stream>>>(x, weight, out, dim, eps);
}

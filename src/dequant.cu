// Q8_0 dequant kernel. One CUDA block per Q8_0 block, one thread per value.
// x = d * q  (Q8_0 has no zero-point offset). See PLAN.md section 11.
#include "dequant.h"

__global__ void dequant_q8_0_kernel(const BlockQ80* blocks, __half* out, int n_blocks) {
    int b = blockIdx.x;
    if (b >= n_blocks) return;
    int j = threadIdx.x; // 0..QK8_0-1
    __half d = blocks[b].d;
    int8_t q = blocks[b].qs[j];
    out[b * QK8_0 + j] = __hmul(d, __int2half_rn((int)q));
}

void dequant_q8_0(const BlockQ80* d_blocks, __half* d_out, int n_blocks, cudaStream_t stream) {
    if (n_blocks <= 0) return;
    dequant_q8_0_kernel<<<n_blocks, QK8_0, 0, stream>>>(d_blocks, d_out, n_blocks);
}

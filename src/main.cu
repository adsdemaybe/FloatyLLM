// SemiLLM entry point (placeholder — replaced by the Phase 1 loader + scheduler).
// Exists so the CUDA build + CI compile from day one; uses only single-line comments.
#include <cstdio>
#include <cuda_runtime.h>

__global__ void noop() {}

int main() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    // CI runners have no GPU; a clean compile + run is still the goal here.
    printf("SemiLLM placeholder. CUDA devices visible: %d (%s)\n",
           count, cudaGetErrorString(err));
    return 0;
}

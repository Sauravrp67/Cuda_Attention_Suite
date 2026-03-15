#include <cstdio>
#include <cuda_runtime.h>
#include "cuda_check.cuh"

int main() {

    int *d_ptr;

    // This should succeed
    CUDA_CHECK(cudaMalloc(&d_ptr, 100 * sizeof(int)));

    printf("cudaMalloc succeeded\n");

    // Intentionally create an error
    CUDA_CHECK(cudaMemcpy(NULL, d_ptr, 100 * sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(d_ptr);

    return 0;
}
#include <cuda_runtime.h>
#include <cstdio>

#ifndef CUDA_CHECK
#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
  std::exit(1); } } while (0)
#endif

// __global__ void strided_read_kernel(const float* __restrict__ in,
//                                     float* __restrict__ out,
//                                     size_t N, int stride)
// {
//     const size_t t  = blockIdx.x * blockDim.x + threadIdx.x;
//     const size_t T  = gridDim.x * (size_t)blockDim.x;

//     float acc = 0.f;

//     for (size_t j = (size_t)t * (size_t)stride; j < N; j += (size_t)T * (size_t)stride) {
//         // across a warp, addresses differ by (stride * sizeof(float))
//         // float v = in[j]; // perfectly coalesced for stride == 1
//         // acc = acc * 1.000000119f + v;  // keep compiler from optimizing out loads
//         float4 v = reinterpret_cast<const float4*>(in)[j >> 2];
//         acc += v.x + v.y + v.z + v.w;
//     }

//     // do one write per thread (no stride so coalesced, negligible vs reads)
//     if (t < N) out[t] = acc;
// }

__global__ void strided_read_kernel(const float4* __restrict__ in,
                                    float* __restrict__ out,
                                    size_t N4)
{
    const size_t t  = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t T  = gridDim.x * (size_t)blockDim.x;

    float4 acc0 = {0,0,0,0};
    float4 acc1 = {0,0,0,0};
    float4 acc2 = {0,0,0,0};
    float4 acc3 = {0,0,0,0};
    
    #pragma unroll
    for (size_t j = t; j < N4; j += T * 4) {

        float4 v0 = in[j];
        float4 v1 = in[j + T];
        float4 v2 = in[j + 2*T];
        float4 v3 = in[j + 3*T];

        acc0.x += v0.x; acc0.y += v0.y; acc0.z += v0.z; acc0.w += v0.w;
        acc1.x += v1.x; acc1.y += v1.y; acc1.z += v1.z; acc1.w += v1.w;
        acc2.x += v2.x; acc2.y += v2.y; acc2.z += v2.z; acc2.w += v2.w;
        acc3.x += v3.x; acc3.y += v3.y; acc3.z += v3.z; acc3.w += v3.w;
    }

    float sum =
        acc0.x + acc0.y + acc0.z + acc0.w +
        acc1.x + acc1.y + acc1.z + acc1.w +
        acc2.x + acc2.y + acc2.z + acc2.w +
        acc3.x + acc3.y + acc3.z + acc3.w;

    out[t] = sum;
}

static float run_case(const float* d_in, float* d_out, size_t N, int stride, int iters, dim3 grid, dim3 block){

    strided_read_kernel<<<grid,block>>>(reinterpret_cast<const float4*>(d_in), d_out, N >> 2);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < iters; ++i) {
        strided_read_kernel<<<grid,block>>>(reinterpret_cast<const float4*>(d_in),d_out, N >> 2);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms,start,stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // bytes read ~= (N/stride) floats per kernel launch
    const double bytes_read = (double)iters * (double)(N / (size_t)stride) * sizeof(float);
    const double sec = ms / 1e3;
    const double GBps = bytes_read / sec / 1e9;
    return (float)GBps;

}


int main() {
        // needs to be big enough to exceed cache size
    const size_t N = (size_t)1 << 26; // 67,108,864 floats ~= 256 MB
    const int    iters = 10;

    // choose sensible launch parameters
    const int block_size = 256;
    int dev = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    const int blocks_per_sm = 8;
    const int grid_size = blocks_per_sm * prop.multiProcessorCount;

    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));

    // initialize input
    CUDA_CHECK(cudaMemset(d_in,  0, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_out, 0, N * sizeof(float)));

    printf("# Device: %s (SM %d)\n", prop.name, prop.major * 10 + prop.minor);
    printf("# N = %zu floats (%.1f MB), iters = %d\n", N, N * sizeof(float) / (1024.0*1024.0), iters);
    printf("%6s  %10s\n", "stride", "GB/s");

    // run multiple strides
    const int strides[] = {1,2,4,8,16,32,64,128,256,512,1024,2048};
    for (int s : strides) {
        // (N ÷ stride) must be an integer
        if ((N % (size_t)s) != 0) continue;
        float gbps = run_case(d_in, d_out, N, s, iters, dim3(grid_size), dim3(block_size));
        printf("%6d  %10.1f\n", s, gbps);
    }

    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_in));
    return 0;
}
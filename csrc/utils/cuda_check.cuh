#pragma once
#include <cuda_runtime.h>
#include <cstdio>

#define CUDA_CHECK(call) \
    if((call) != cudaSuccess) { \
        printf("CUDA Error: %s at line %d\n", cudaGetErrorString(cudaGetLastError()), __LINE__); \
        exit(1); \
    }
    
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cuda_attention_suite LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


# This command queries the GPU driver directly
message(STATUS "Querying GPU Architecture from Driver")
execute_process(
    COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits
    OUTPUT_VARIABLE GPU_ARCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
# Remove decimal point (e.g., 8.6 -> 86)
string(REPLACE "." "" GPU_ARCH_CLEAN ${GPU_ARCH})
set(CMAKE_CUDA_ARCHITECTURES ${GPU_ARCH})

set(TORCH_CUDA_ARCH_LIST "${CMAKE_CUDA_ARCHITECTURES}")

message(STATUS "Targeting CUDA Architectures: sm_${GPU_ARCH_CLEAN}")

# Standard C++ and CUDA flags for build process
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --use_fast_math -Xptxas=-v -O3")

# Locate PyTorch(LibTorch)
find_package(Torch REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Define the extension Module
add_library(cuda_attn_backend SHARED
    csrc/bindings.cpp
    csrc/attention/naive_v1/naive_kernel.cu
)

# Ensure we can find out our own headers
target_include_directories(cuda_attn_backend PRIVATE csrc)
target_include_directories(cuda_attn_backend PRIVATE ${Python3_INCLUDE_DIRS})

# Link Torch with internal libraries (ATen)
target_link_libraries(cuda_attn_backend PRIVATE "${TORCH_LIBRARIES}")
target_link_libraries(cuda_attn_backend PRIVATE "${Python3_LIBRARIES}")

# Naming Conventions to match python expectations
set_target_properties(cuda_attn_backend PROPERTIES PREFIX "")
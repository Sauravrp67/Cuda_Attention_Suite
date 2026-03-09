cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(Attention_Variants LANGUAGES CXX CUDA)

# -----------------------------------------------------------------------------
# 1. High-Performance Hardware Detection
# -----------------------------------------------------------------------------

execute_process(
    COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits
    OUTPUT_VARIABLE GPU_ARCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REPLACE "." "" GPU_ARCH_CLEAN ${GPU_ARCH})
set(CMAKE_CUDA_ARCHITECTURES ${GPU_ARCH})

message(STATUS "🚀 Targeting CUDA Architecture: sm_${GPU_ARCH_CLEAN}")

# -----------------------------------------------------------------------------
# 2. Dependencies & Environment Sync
# -----------------------------------------------------------------------------
find_package(Torch REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Automatically match the ABI of the installed PyTorch (Fixes the ImportError)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
    OUTPUT_VARIABLE TORCH_ABI
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "🛠️ Syncing with PyTorch ABI: ${TORCH_ABI}")

# -----------------------------------------------------------------------------
# 3. Compiler Global Configurations
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# -----------------------------------------------------------------------------
# 4. Define the Extension Library
# -----------------------------------------------------------------------------
add_library(cuda_attn_backend SHARED
    csrc/bindings.cpp
    csrc/attention/naive_v1/naive_kernel.cu
)

# -----------------------------------------------------------------------------
# 5. Target-Specific Properties (The "Full Depth" Way)
# -----------------------------------------------------------------------------
target_compile_definitions(cuda_attn_backend PRIVATE 
    _GLIBCXX_USE_CXX11_ABI=${TORCH_ABI}
)

target_compile_options(cuda_attn_backend PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall>
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --use_fast_math -Xptxas=-v -O3 -Xcompiler -fPIC>
)

# Crucial: Pass the ABI flag to the NVCC host compiler (gcc)
target_compile_options(cuda_attn_backend PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -D_GLIBCXX_USE_CXX11_ABI=${TORCH_ABI}>
)

# -----------------------------------------------------------------------------
# 6. Include Paths & Linking
# -----------------------------------------------------------------------------
target_include_directories(cuda_attn_backend PRIVATE 
    csrc 
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(cuda_attn_backend PRIVATE 
    ${TORCH_LIBRARIES} 
    ${Python3_LIBRARIES}
)

# -----------------------------------------------------------------------------
# 7. Output & Naming (Match Python Import Requirements)
# -----------------------------------------------------------------------------
set_target_properties(cuda_attn_backend PROPERTIES 
    PREFIX "" 
    SUFFIX ".so"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Custom message for developer feedback
message(STATUS "✅ Build configured. Shared object will be in: ${CMAKE_BINARY_DIR}/lib")
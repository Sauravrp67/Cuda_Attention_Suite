cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(Attention_Variants LANGUAGES CXX CUDA)

# -----------------------------------------------------------------------------
# 1. CUDA GPU Detection
# -----------------------------------------------------------------------------

execute_process(
    COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits
    OUTPUT_VARIABLE GPU_ARCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REPLACE "\n" ";" GPU_ARCH_LIST "${GPU_ARCH}")
list(GET GPU_ARCH_LIST 0 GPU_ARCH)
string(REPLACE "." "" GPU_ARCH_CLEAN ${GPU_ARCH})
set(CMAKE_CUDA_ARCHITECTURES ${GPU_ARCH})

message(STATUS "Targeting CUDA Architecture: sm_${GPU_ARCH_CLEAN}")

# Lock Pytorch compilation architector to local Device Architecture
set(TORCH_CUDA_ARCH_LIST ${CMAKE_CUDA_ARCHITECTURES})
message(STATUS "Locked to Architecture: sm_${CMAKE_CUDA_ARCHITECTURES}")

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

# Link libtorch_python.so with target for including PyObj to at::Tensor casting logic
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.__path__[0])"
    OUTPUT_VARIABLE TORCH_LIB_DIR 
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import sys; import os; print(os.path.join(sys.prefix, 'lib'))"
    OUTPUT_VARIABLE PYTHON_LIB_DIR
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

option(BUILD_LEGACY_PYCUATTENTION "Build the legacy PyCuAttention extension target." OFF)
set(
    ATTENTION_VARIANTS_OUTPUT_DIR
    "${CMAKE_SOURCE_DIR}/src/attention_variants/backends/cuda"
    CACHE PATH
    "Output directory for the structured attention_variants CUDA extension."
)
set(
    ATTENTION_VARIANTS_OUTPUT_NAME
    "attention_variants_cuda"
    CACHE STRING
    "Basename of the structured attention_variants CUDA extension."
)

# -----------------------------------------------------------------------------
# 4. Shared Extension Helper
# -----------------------------------------------------------------------------
function(add_torch_cuda_extension target_name output_name output_dir)
    add_library(${target_name} SHARED ${ARGN})

    target_compile_options(${target_name} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall>
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --use_fast_math -Xptxas=-v -O3 -Xcompiler -fPIC>
    )

    target_include_directories(${target_name} PRIVATE
        csrc
        ${Python3_INCLUDE_DIRS}
        ${TORCH_INCLUDE_DIRS}
    )

    target_link_libraries(${target_name} PRIVATE
        ${TORCH_LIBRARIES}
        ${Python3_LIBRARIES}
        "${TORCH_LIB_DIR}/lib/libtorch_python.so"
    )

    set_target_properties(${target_name} PROPERTIES
        OUTPUT_NAME "${output_name}"
        PREFIX ""
        SUFFIX ".so"
        LIBRARY_OUTPUT_DIRECTORY "${output_dir}"
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "${TORCH_LIB_DIR}/lib/;${PYTHON_LIB_DIR}"
    )
endfunction()

# -----------------------------------------------------------------------------
# 5. Backend Source Sets
# -----------------------------------------------------------------------------

set(LEGACY_BACKEND_SOURCES
    csrc/bindings.cpp
    csrc/attention/naive_v1/naive_kernel.cu
    csrc/linalg/matmul.cu
)

set(STRUCTURED_BACKEND_SOURCES
    csrc/bindings/runtime.cpp
    csrc/bindings/primitives.cpp
    csrc/bindings/attention_core.cpp
    csrc/attention/core/sdpa/naive/naive_kernel.cu
    csrc/primitives/matmul/matmul.cu
)

# -----------------------------------------------------------------------------
# 6. Build the Legacy and Structured Python Extensions
# -----------------------------------------------------------------------------

message("Torch Include DIR: ${TORCH_INCLUDE_DIRS}")
message("Python3 Include DIR: ${Python3_INCLUDE_DIRS}")

if(BUILD_LEGACY_PYCUATTENTION)
    add_torch_cuda_extension(cuda_attn_backend cuda_attn_backend "${CMAKE_BINARY_DIR}/lib"
        ${LEGACY_BACKEND_SOURCES}
    )
endif()

add_torch_cuda_extension(
    attention_variants_cuda
    "${ATTENTION_VARIANTS_OUTPUT_NAME}"
    "${ATTENTION_VARIANTS_OUTPUT_DIR}"
    ${STRUCTURED_BACKEND_SOURCES}
)

message(STATUS "Structured extension output directory: ${ATTENTION_VARIANTS_OUTPUT_DIR}")
if(BUILD_LEGACY_PYCUATTENTION)
    message(STATUS "Legacy extension output directory: ${CMAKE_BINARY_DIR}/lib")
endif()

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

# -----------------------------------------------------------------------------
# 4. Define the Extension Library
# -----------------------------------------------------------------------------
add_library(cuda_attn_backend SHARED
    csrc/bindings.cpp
    csrc/attention/naive_v1/naive_kernel.cu
)

# -----------------------------------------------------------------------------
# 5. Target-Specific Properties
# -----------------------------------------------------------------------------
# target_compile_definitions(cuda_attn_backend PRIVATE 
#     _GLIBCXX_USE_CXX11_ABI=${TORCH_ABI}
# )

target_compile_options(cuda_attn_backend PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall>
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --use_fast_math -Xptxas=-v -O3 -Xcompiler -fPIC>
)


# target_compile_options(cuda_attn_backend PRIVATE
#     $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -D_GLIBCXX_USE_CXX11_ABI=${TORCH_ABI}>
# )

# -----------------------------------------------------------------------------
# 6. Include Paths & Linking
# -----------------------------------------------------------------------------

message("Torch Include DIR: ${TORCH_INCLUDE_DIRS}")
message("Python3 Include DIR: ${Python3_INCLUDE_DIRS}")

target_include_directories(cuda_attn_backend PRIVATE 
    csrc 
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(cuda_attn_backend PRIVATE 
    ${TORCH_LIBRARIES} 
    ${Python3_LIBRARIES}
    "${TORCH_LIB_DIR}/lib/libtorch_python.so"
    )

# -----------------------------------------------------------------------------
# 7. Output & Naming (Match Python Import Requirements)
# -----------------------------------------------------------------------------
set_target_properties(cuda_attn_backend PROPERTIES 
    PREFIX "" 
    SUFFIX ".so"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${TORCH_LIB_DIR}/lib/;${PYTHON_LIB_DIR}"
)

message(STATUS "Build configured. Shared object will be in: ${CMAKE_BINARY_DIR}/lib")
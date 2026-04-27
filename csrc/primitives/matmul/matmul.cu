#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>

#include "primitives/matmul/matmul.h"

#define TILE_WIDTH 32

// smem TILES Dimensions
#define TILE_WIDTH_M 32
#define TILE_WIDTH_K 32
#define TILE_WIDTH_N 32

// reg TILES Dimension per thread
#define THREAD_M 2
#define THREAD_N 2

template <typename scalar_t>
__global__ void naive_gemm(
    const scalar_t* __restrict__ A, const int M, const int K, const int lda, bool transA,
    const scalar_t* __restrict__ B, const int N, const int ldb, bool transB,
    scalar_t* __restrict__ C, int ldc,
    const float alpha, float beta
) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int column = blockDim.x * blockIdx.x + threadIdx.x;

    float p_value = 0.0f;
    if (row < M && column < N) {
        for (int k = 0; k < K; k++) {
            // Each thread maps to the each element of output buffer. 
            float a_value = transA ? (float)A[k * lda + row] : (float)A[row * lda + k];
            float b_value = transB ? (float)B[column * ldb + k] : (float)B[k * ldb + column];
            
            p_value += a_value * b_value;
        }
        const float c_value = beta == 0.0f ? 0.0f : static_cast<float>(C[row * ldc + column]);
        C[row * ldc + column] = scalar_t(alpha * p_value + beta * c_value);
    }
}

template<typename scalar_t>
__global__ void tiled_gemm(const scalar_t* A, const int M, const int K, const int lda, bool transA,
        const scalar_t* B, const int N, const int ldb, bool transB,
        scalar_t* C, const int ldc,
        const float alpha, const float beta) {

            int bx = blockIdx.x;
            int by = blockIdx.y;
            int tx = threadIdx.x;
            int ty = threadIdx.y;

            __shared__ scalar_t Mds[TILE_WIDTH][TILE_WIDTH];
            __shared__ scalar_t Nds[TILE_WIDTH][TILE_WIDTH];
            
            int row = by * blockDim.y + ty;
            int column = bx * blockDim.x + tx;
            
            float pValue = 0.0f;

            for(size_t ph = 0; ph < (K + TILE_WIDTH - 1) / TILE_WIDTH; ph++) {

                if(row < M && (ph * TILE_WIDTH + tx) < K) {
                    Mds[ty][tx] = transA
                        ? A[(ph * TILE_WIDTH + tx) * lda + row]
                        : A[row * lda + ph * TILE_WIDTH + tx];
                }
                else Mds[ty][tx] = 0.0f;
                
                if(column < N && (ph * TILE_WIDTH + ty) < K) {
                    Nds[ty][tx] = transB 
                        ? B[column * ldb + ph *TILE_WIDTH + ty] 
                        : B[(ph * TILE_WIDTH + ty) * ldb + column];
                }
                else Nds[ty][tx] = 0.0f;

                __syncthreads();

                for(size_t k = 0; k < TILE_WIDTH; k++) {
                    float Mds_value = static_cast<float>(Mds[ty][k]);
                    float Nds_value = static_cast<float>(Nds[k][tx]);
                    pValue += Mds_value * Nds_value;
                }
                __syncthreads();
            }

        if (row < M && column < N) {
            const float cValue = beta == 0.0f ? 0.0f : static_cast<float>(C[row * ldc + column]);
            C[row * ldc + column] = (scalar_t)(alpha * pValue + beta * cValue);
        }
}

template<typename scalar_t>
__global__ void var_tiled_gemm(const scalar_t* A, const int M, const int K, const int lda, bool transA,
        const scalar_t* B, const int N, const int ldb, bool transB,
        scalar_t* C, const int ldc,
        const float alpha, const float beta) {

            const uint bx{blockIdx.x}; const uint by{blockIdx.y};
            const uint tx{threadIdx.x}; const uint ty{threadIdx.y};

            __shared__ scalar_t Mds[TILE_WIDTH_M][TILE_WIDTH_K];
            __shared__ scalar_t Nds[TILE_WIDTH_K][TILE_WIDTH_N];

            const uint row{by * blockDim.y + ty};
            const uint column{bx * blockDim.x + tx};
            
            float pValue = 0.0f;

            for(size_t ph = 0; ph < (K + TILE_WIDTH_K - 1 ) / TILE_WIDTH_K; ph++) {

                // Load Block Tiles of A in SMEM
                for (size_t i = tx; i < TILE_WIDTH_K; i += blockDim.x){
                    for(size_t j = ty; j < TILE_WIDTH_M; j += blockDim.y) {

                        if(row < M && (ph * TILE_WIDTH_K + i) < K){
                            Mds[j][i] = transA
                                ? A[(ph * TILE_WIDTH_K + i) * lda + row]
                                : A[row * lda + ph * TILE_WIDTH_K + i];
                        }
                        else Mds[j][i] = 0.0f;
                    }
                }
                
                //Load Block Tiles of B in SMEM
                for(size_t i = ty; i < TILE_WIDTH_K; i += blockDim.y){
                    for(size_t j = tx; j< TILE_WIDTH_N; j += blockDim.x) {
                        if(column < N && (ph * TILE_WIDTH_K + i) < K) {
                            Nds[i][j] = transB
                                ? B[column * ldb + (ph * TILE_WIDTH_K + i)]
                                : B[(ph * TILE_WIDTH_K + i) * ldb + column]; 
                        }
                        else Nds[i][j] = 0.0f;
                    }
                }
                __syncthreads();
                
                for(size_t k = 0; k < TILE_WIDTH_K; k++){

                    float Mds_value = static_cast<float>(Mds[ty][k]);
                    float Nds_value = static_cast<float>(Nds[k][tx]);

                    pValue += Mds_value * Nds_value;

                    }
                __syncthreads();
            }

            if (row < M && column < N) {
                const float cValue = beta == 0.0f ? 0.0f : static_cast<float>(C[row * ldc + column]);
                C[row * ldc + column] = (scalar_t)(alpha * pValue + beta * cValue);
            }        
        }


template<typename scalar_t>
__global__ void register2DTiledSgemm(const scalar_t* A, const int M, const int K, const int lda, bool transA,
        const scalar_t* B, const int N, const int ldb, bool transB,
        scalar_t* C, const int ldc,
        const float alpha, const float beta) {
            
        const size_t bx{blockIdx.x}; const size_t by{blockIdx.y};
        const size_t tx{threadIdx.x}; const size_t ty{threadIdx.y};

        __shared__ scalar_t Mds[TILE_WIDTH_M][TILE_WIDTH_K];
        __shared__ scalar_t Nds[TILE_WIDTH_K][TILE_WIDTH_N];

        const size_t row{by * TILE_WIDTH_M + ty * THREAD_M}; const size_t column{bx * TILE_WIDTH_N + tx * THREAD_N};
        
        size_t global_row, global_col,global_k, global_k2;

        float PVALUES[THREAD_M][THREAD_N] = {0.0f};

        for(size_t ph = 0; ph < (K + TILE_WIDTH_K - 1) / TILE_WIDTH_K; ph++) {
            
            //Load TILE_M X TILE_K per block in smem
            for(size_t i = tx; i < TILE_WIDTH_K; i += blockDim.x){
                
                for(size_t j = ty; j < TILE_WIDTH_M; j += blockDim.y) {
                    
                    global_row = by * TILE_WIDTH_M + j;
                    global_k = ph * TILE_WIDTH_K + i;

                    if(global_row < M && global_k < K){    
                        Mds[j][i] = transA
                            ? A[global_k * lda + global_row]
                            : A[global_row * lda + global_k];
                    }
                    else Mds[j][i] = 0.0f;
            }
        }

            // Load TILE_K * TILE_N per block in smem
            for(size_t i = ty; i < TILE_WIDTH_K; i += blockDim.y){
                for(size_t j = tx; j < TILE_WIDTH_N; j += blockDim.x){
                    
                    global_col = bx * TILE_WIDTH_N + j;
                    global_k2 = ph * TILE_WIDTH_K + i;
                    
                    if(global_col < N && global_k2 < K) {
                        Nds[i][j] = transB
                            ? B[(global_col * ldb + global_k2)]
                            : B[global_k2 * ldb + global_col]; 
                    }
                    else Nds[i][j] = 0.0f;
                }
            }
            __syncthreads();

            for(size_t k = 0; k < TILE_WIDTH_K; k++) {
                scalar_t a_reg[THREAD_M];
                scalar_t b_reg[THREAD_N];

                // Load 2 x 1 TILE into a_reg
                for (size_t i = 0; i < THREAD_M; i++){
                    a_reg[i] = Mds[ty * THREAD_M + i][k];
                }

                // Load 1 x 2 TILE into b_reg
                for(size_t j = 0; j < THREAD_N;j++){
                    b_reg[j] = Nds[k][tx * THREAD_N + j];
                }

                for(size_t i = 0; i < THREAD_M; i++){
                    for(size_t j = 0; j < THREAD_N; j++){
                        PVALUES[i][j] += static_cast<float>(a_reg[i]) * static_cast<float>(b_reg[j]);
                    }
                }
            }
            __syncthreads();
        }
    
    // Write PVALUES TILE to Global Memory
    for(size_t i = 0; i < THREAD_M; i++){
        for(size_t j = 0; j < THREAD_N; j++){
            if((row + i) < M && (column + j) < N){
                const float c_value = beta == 0.0f ? 0.0f : static_cast<float>(C[(row + i) * ldc + (column +j)]);
                C[(row + i) * ldc + (column + j)] = (scalar_t)(alpha * PVALUES[i][j] + beta * c_value);
            }
        }
    }
    
}
                                    

struct GemmProblem {
    const at::Tensor& input;
    const at::Tensor& weight;
    at::Tensor output;
    int M;
    int K;
    int N;
    int lda;
    int ldb;
    int ldc;
    bool transA;
    bool transB;
    float alpha;
    float beta;
};

static GemmProblem make_gemm_problem(
    const at::Tensor& input,
    const at::Tensor& weight,
    bool transA,
    bool transB
) {
    TORCH_CHECK(input.is_cuda(), "GEMM input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "GEMM weight must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "GEMM input must be 2D, got ", input.dim(), "D");
    TORCH_CHECK(weight.dim() == 2, "GEMM weight must be 2D, got ", weight.dim(), "D");
    TORCH_CHECK(
        input.scalar_type() == weight.scalar_type(),
        "GEMM input and weight must have the same dtype"
    );
    TORCH_CHECK(input.device() == weight.device(), "GEMM tensors must be on the same device");
    TORCH_CHECK(input.is_contiguous(), "GEMM input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "GEMM weight must be contiguous");

    const int input_rows = input.size(0);
    const int input_cols = input.size(1);
    const int weight_rows = weight.size(0);
    const int weight_cols = weight.size(1);

    const int M = transA ? input_cols : input_rows;
    const int K = transA ? input_rows : input_cols;

    const int weight_K = transB ? weight_cols : weight_rows;
    const int N = transB ? weight_rows : weight_cols;

    const int lda = input_cols;
    const int ldb = weight_cols;

    TORCH_CHECK(
        K == weight_K,
        "Shape mismatch for GEMM: input K (", K,
        ") must match weight K (", weight_K, ")"
    );

    return GemmProblem{
        input,
        weight,
        at::empty({M, N}, input.options()),
        M,
        K,
        N,
        lda,
        ldb,
        N,
        transA,
        transB,
        1.0f,
        0.0f,
    };
}

struct NaiveGemmLauncher {
    static constexpr int kThreadsX = 16;
    static constexpr int kThreadsY = 16;
    static constexpr const char* kDispatchName = "naive_gemm_structured";

    template <typename scalar_t>
    static void launch(const GemmProblem& problem, dim3 grid, dim3 block) {
        naive_gemm<scalar_t><<<grid, block>>>(
            problem.input.data_ptr<scalar_t>(),
            problem.M,
            problem.K,
            problem.lda,
            problem.transA,
            problem.weight.data_ptr<scalar_t>(),
            problem.N,
            problem.ldb,
            problem.transB,
            problem.output.data_ptr<scalar_t>(),
            problem.ldc,
            problem.alpha,
            problem.beta
        );
    }
};

struct TiledGemmLauncher {
    static constexpr int kThreadsX = 32;
    static constexpr int kThreadsY = 32;
    static constexpr const char* kDispatchName = "tiled_gemm_structured";

    template <typename scalar_t>
    static void launch(const GemmProblem& problem, dim3 grid, dim3 block) {
            tiled_gemm<scalar_t><<<grid, block>>>(
                problem.input.data_ptr<scalar_t>(),
                problem.M,
                problem.K,
                problem.lda,
                problem.transA,
                problem.weight.data_ptr<scalar_t>(),
                problem.N,
                problem.ldb,
                problem.transB,
                problem.output.data_ptr<scalar_t>(),
                problem.ldc,
                problem.alpha,
                problem.beta
            );
        }
};

struct VarTiledGemmLauncher {
    static constexpr int kThreadsX = 32;
    static constexpr int kThreadsY = 32;
    static constexpr const char* kDispatchName = "var_tiled_gemm_structured";

    template <typename scalar_t>
    static void launch(const GemmProblem& problem, dim3 grid, dim3 block) {
            var_tiled_gemm<scalar_t><<<grid, block>>>(
                problem.input.data_ptr<scalar_t>(),
                problem.M,
                problem.K,
                problem.lda,
                problem.transA,
                problem.weight.data_ptr<scalar_t>(),
                problem.N,
                problem.ldb,
                problem.transB,
                problem.output.data_ptr<scalar_t>(),
                problem.ldc,
                problem.alpha,
                problem.beta
            );
        }
};

struct regTiledsgemmLauncher {
    static constexpr int kThreadsX = 16;
    static constexpr int kThreadsY = 16;
    static constexpr const char* kDispatchName = "Reg_Tiled_gemm_structured";

    template <typename scalar_t>
    static void launch(const GemmProblem& problem, dim3 grid, dim3 block) {
            register2DTiledSgemm<scalar_t><<<grid, block>>>(
                problem.input.data_ptr<scalar_t>(),
                problem.M,
                problem.K,
                problem.lda,
                problem.transA,
                problem.weight.data_ptr<scalar_t>(),
                problem.N,
                problem.ldb,
                problem.transB,
                problem.output.data_ptr<scalar_t>(),
                problem.ldc,
                problem.alpha,
                problem.beta
            );
        }
};

template <typename Launcher>
static at::Tensor dispatch_gemm(
    const at::Tensor& input,
    const at::Tensor& weight,
    bool transA,
    bool transB
) {
    GemmProblem problem = make_gemm_problem(input, weight, transA, transB);

    dim3 block(Launcher::kThreadsX, Launcher::kThreadsY, 1);
    dim3 grid(
        (problem.N + Launcher::kThreadsX - 1) / Launcher::kThreadsX,
        (problem.M + Launcher::kThreadsY - 1) / Launcher::kThreadsY,
        1
    );

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        Launcher::kDispatchName,
        [&] {
            Launcher::template launch<scalar_t>(problem, grid, block);
        }
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return problem.output;
}

at::Tensor gemm(const at::Tensor& input, const at::Tensor& weight, bool transA, bool transB) {
    return dispatch_gemm<NaiveGemmLauncher>(input, weight, transA, transB);
}

at::Tensor sgemm(const at::Tensor& input, const at::Tensor& weight, bool transA, bool transB) {
    return dispatch_gemm<TiledGemmLauncher>(input, weight, transA, transB);
}

at::Tensor var_sgemm(const at::Tensor& input, const at::Tensor& weight, bool transA, bool transB) {
    return dispatch_gemm<VarTiledGemmLauncher>(input, weight, transA, transB);
}

at::Tensor reg2DTiledsgemm(const at::Tensor& input, const at::Tensor& weight, bool transA, bool transB) {
    return dispatch_gemm<regTiledsgemmLauncher>(input,weight,transA,transB);
}

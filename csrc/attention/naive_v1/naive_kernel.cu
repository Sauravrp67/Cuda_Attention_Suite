#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>

#define MAX_SEQ_LEN 4096

template<typename scalar_t>
__global__ void naive_attention_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    scalar_t* __restrict__ o,
    const int B, const int H, const int N, const int D){
        
        // Thread Indexing
        int b = blockIdx.z;
        int h = blockIdx.y;
        int i = blockIdx.x * blockDim.x + threadIdx.x; 
        float scores[MAX_SEQ_LEN];
        const float scale = rsqrtf((float)D);

        if (i < N) {
            // Calculate QK^T matrix:
            for (int j = 0; j < N; j++) {
                float score = 0;
                for (int d = 0; d < D; d++) {
                    score += (float)q[b * H * N * D + h * N * D + i * D + d] * (float)k[b * H * N * D + h * N * D + j * D + d];
                }
                scores[j] = score * scale;
            }

            //Offline Softmax
            // Calculation of Max attention score over a row:
            float max = scores[0];
            for(int j = 0; j < N; j++) {
                if (scores[j] > max) max = scores[j];
            }

            //Calculation of Sum
            float sum = 0;
            for(int j = 0; j < N; j++) {
                scores[j] = expf(scores[j] - max);
                sum += scores[j];
            }

            // Calculation of softmax
            for (int j = 0; j < N; j++) {
                scores[j] /= sum;
            }
            
            /*Calculation of softmax(QK^T)/sqrt(D))V 
             Here we are doing accumulation over D times for a single accumulation: meaning float32 multiply and float16/bfloat16 addition N times.
             So Rounding Error occurs N times on each += to o[...]. If we do a outer D loop and inner N loop with accumulation happening on register by each thread,
             We incur rounding error for accumulation only 1 time per element in the output matrix. 
            */

            //Outer N loop(token) and inner D loop(head dimension)
            // for (int j = 0; j < N; j++){
            //     for (int d = 0; d < D; d++) {
            //         // Here we are performing N*D read and write to global memory. It is costly. We ought to instead store this result in 
            //         // Another buffer like: float weight_acc[D]; but this eats up the register, if D is large like  
            //         o[b * H * N * D + h * N * D + i * D + d] += (scalar_t)(scores[j] * (float)v[b * H * N * D + h * N * D + j * D + d]);
            //     }
            // }

            //Outer D loop(head dimension) and inner N loop(token).
            for (int d = 0; d < D; d++) {
                float acc = 0;
                for (int j = 0; j < N; j++) {
                    acc += scores[j] * (float)v[b * H * N * D + h * N * D + j * D + d];
                }
                o[b * H * N * D + h * N * D + i * D + d] = (scalar_t) acc;
            }
        }
    };

at::Tensor naive_attention_fwd(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v) {

    const int B = q.size(0);
    const int H = q.size(1);
    const int N = q.size(2);
    const int D = q.size(3);

    auto opts = q.options();

    at::Tensor out = torch::zeros_like(q,opts);

    dim3 grid((N + 256 -1 )/256, H, B);
    dim3 block(256);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        q.scalar_type(),
        "native_attention_fwd",
        [&]() {
            naive_attention_kernel<scalar_t><<<grid, block>>>(
                q.data_ptr<scalar_t>(),
                k.data_ptr<scalar_t>(),
                v.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                B, H, N, D
            );
        }
    );
    
    return out;
}
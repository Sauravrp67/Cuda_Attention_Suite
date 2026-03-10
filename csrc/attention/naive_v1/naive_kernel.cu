#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void naive_attention_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ o,
    const int B, const int H, const int N, const int D,const float scaling){
        
        //Thread Indexing
        int b = blockIdx.z;
        int h = blockIdx.y;
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        
        // each thread has unique "i", each i represents a unique token
        if(i < N) {
            // Compute: QK^T
            // This represents each unique token in "K", vector. That is each "j", is a unique token in k vector.
            for (int j = 0; j < N; ++j) {
                float score = 0.0f;
                // This represents head dimension. That is each "d" is a Q and K value of each token.
                for(int d = 0; d < D; ++d) {
                    score += q[b * H * N * D + h * N * D + i * D + d] * k[b * H * N * D + h * N * D + j * D + d];
                }
            score *= scaling;

            // Compute: QK^TV
            for( int d = 0; d < D; ++d) {
                atomicAdd(&o[b * H * N * D + h * N * D + i * D + d],score * v[b * H * N * D + h * N * D + j * D + d]);
            }
            }
        }
    };

at::Tensor naive_attention_fwd(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, float scaling) {

    const int B = q.size(0);
    const int H = q.size(1);
    const int N = q.size(2);
    const int D = q.size(3);

    auto opts = q.options();

    at::Tensor out = torch::zeros_like(q,opts);

    dim3 grid((N + 256 -1 )/256, H, B);
    dim3 block(256);

    naive_attention_kernel<<<grid,block>>>(
                                            q.data_ptr<float>(),
                                            k.data_ptr<float>(),
                                            v.data_ptr<float>(),
                                            out.data_ptr<float>(),
                                            B,H,N,D,scaling);

    return out;
}
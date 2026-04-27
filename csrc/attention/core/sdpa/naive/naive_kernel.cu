#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>

#include "attention/core/sdpa/naive/naive_fwd.h"

#define MAX_SEQ_LEN 4096

template <typename scalar_t>
__global__ void naive_attention_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    scalar_t* __restrict__ o,
    const int B, const int H, const int N, const int D
) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float scores[MAX_SEQ_LEN];
    const float scale = rsqrtf((float)D);

    if (i < N) {
        for (int j = 0; j < N; j++) {
            float score = 0.0f;
            for (int d = 0; d < D; d++) {
                score +=
                    (float)q[b * H * N * D + h * N * D + i * D + d] *
                    (float)k[b * H * N * D + h * N * D + j * D + d];
            }
            scores[j] = score * scale;
        }

        float max_score = scores[0];
        for (int j = 0; j < N; j++) {
            if (scores[j] > max_score) {
                max_score = scores[j];
            }
        }

        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            scores[j] = expf(scores[j] - max_score);
            sum += scores[j];
        }

        const float inv_sum = 1.0f / sum;
        for (int j = 0; j < N; j++) {
            scores[j] *= inv_sum;
        }

        for (int d = 0; d < D; d++) {
            float acc = 0.0f;
            for (int j = 0; j < N; j++) {
                acc += scores[j] * (float)v[b * H * N * D + h * N * D + j * D + d];
            }
            o[b * H * N * D + h * N * D + i * D + d] = (scalar_t)acc;
        }
    }
}

at::Tensor naive_attention_fwd(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v
) {
    const int B = q.size(0);
    const int H = q.size(1);
    const int N = q.size(2);
    const int D = q.size(3);

    at::Tensor out = torch::empty_like(q, q.options());

    dim3 grid((N + 256 - 1) / 256, H, B);
    dim3 block(256);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        q.scalar_type(),
        "naive_attention_fwd_structured",
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

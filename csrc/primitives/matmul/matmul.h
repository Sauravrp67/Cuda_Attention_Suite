#pragma once

#include <torch/extension.h>

at::Tensor gemm(
    const at::Tensor& input,
    const at::Tensor& weight,
    bool transA,
    bool transB
);

at::Tensor sgemm(
    const at::Tensor& input,
    const at::Tensor& weight,
    bool transA,
    bool transB
);

at::Tensor var_sgemm(
    const at::Tensor& input,
    const at::Tensor& weight,
    bool transA,
    bool transB
);

at::Tensor reg2DTiledsgemm(
    const at::Tensor& input, 
    const at::Tensor& weight, 
    bool transA, 
    bool transB
);
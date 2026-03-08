#pragma once
#include <torch/extension.h>

at::Tensor naive_attention_fwd(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    float scaling
);
#include <torch/extension.h>
#include "attention/naive_v1/naive_fwd.h"


PYBIND11_MODULE(cuda_attn_backend, m){
    m.def(
        "naive_attention_fwd", 
        &naive_attention_fwd, 
        "Naive Attention Forward Pass (CUDA)",
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("scaling")
    );
}


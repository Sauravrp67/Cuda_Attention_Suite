#include <torch/extension.h>
#include "attention/naive_v1/naive_fwd.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def(
        "naive_attention_forward", 
        &naive_attention_fwd, 
        "Naive Attention Forward Pass (CUDA)",
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("scaling")
    );
}


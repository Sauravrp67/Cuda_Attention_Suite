// #ifndef _GLIBCXX_USE_CXX11_ABI
// #define _GLIBCXX_USE_CXX11_ABI 1
// #endif

#include <torch/extension.h>
#include "attention/naive_v1/naive_fwd.h"
#include <string>
std::string abi_test(std::string input){
    return input;
}

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

PYBIND11_MODULE(abi_test, m) {
    m.def(
        "abi_test",
        &abi_test,
        "ABI Test",
        py::arg("input")
    );
}



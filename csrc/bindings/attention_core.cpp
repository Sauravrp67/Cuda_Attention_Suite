#include <torch/extension.h>

#include "attention/core/sdpa/naive/naive_fwd.h"
#include "bindings/register.h"

namespace py = pybind11;

void register_attention_core(py::module_& m) {
    m.def(
        "naive_attention_fwd",
        &naive_attention_fwd,
        "Naive attention forward pass (structured backend)",
        py::arg("q"),
        py::arg("k"),
        py::arg("v")
    );
}

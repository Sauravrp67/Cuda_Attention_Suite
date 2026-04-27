#include <torch/extension.h>

#include "bindings/register.h"

PYBIND11_MODULE(attention_variants_cuda, m) {
    register_attention_core(m);
    register_primitives(m);
}

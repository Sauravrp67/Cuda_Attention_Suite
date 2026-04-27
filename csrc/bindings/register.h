#pragma once

#include <pybind11/pybind11.h>

void register_attention_core(pybind11::module_& m);
void register_primitives(pybind11::module_& m);

///////////////////////////////////////////////////////////////////////
// This file is part of the PySYCL software for SYCL development in
// Python. It is licensed under the Apache License, Version 2.0. A copy
// of this license, in a file named LICENSE.md, should have been
// distributed with this file. A copy of this license is also
// currently available at "http://www.apache.org/licenses/LICENSE-2.0".

// Unless explicitly stated, all contributions intentionally submitted
// to this project shall also be under the terms and conditions of this
// license, without any additional terms or conditions.
///////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////
/// \file
/// \brief Python module for tensor in PySYCL.
///////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////
// local
///////////////////////////////////////////////////////////////////////
#include "../Device/Device.h"
#include "Tensor.h"
#include "Tensor_Factories.h"

///////////////////////////////////////////////////////////////////////
// pybind11
///////////////////////////////////////////////////////////////////////
#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

///////////////////////////////////////////////////////////////////////
// stl
///////////////////////////////////////////////////////////////////////
#include <typeinfo>
#include <variant>
#include <vector>

namespace py = pybind11;

///////////////////////////////////////////////////////////////////////
// Declaring types for tensor
///////////////////////////////////////////////////////////////////////
using Device_T = pysycl::Device;
using Data_T = pysycl::Data_Types;
using Module_T = py::module_;

void tensor_base(Module_T& m) {
  m.def(
      "tensor",
      [](Device_T& device, Data_T& dtype) {
        return pysycl::tensor_factories(device, dtype);
      },
      py::arg("device"),
      py::arg("dtype"));
}

void tensor_dims(Module_T& m) {
  m.def(
      "tensor",
      [](Device_T& device, py::tuple& dims, Data_T& dtype) {
        return pysycl::tensor_factories(device, dims, dtype);
      },
      py::arg("device"),
      py::arg("dims"),
      py::arg("dtype"));
}

template<typename Scalar_T>
void tensor_1d(Module_T& m) {
  m.def(
      "tensor",
      [](Device_T& device, std::vector<Scalar_T>& data, Data_T& dtype) {
        return pysycl::tensor_factories(device, data, dtype);
      },
      py::arg("device"),
      py::arg("data"),
      py::arg("dtype"));
}

template<typename Scalar_T>
void tensor_2d(Module_T& m) {
  m.def(
      "tensor",
      [](Device_T& device,
         std::vector<std::vector<Scalar_T>>& data,
         Data_T& dtype) {
        return pysycl::tensor_factories(device, data, dtype);
      },
      py::arg("device"),
      py::arg("data"),
      py::arg("dtype"));
}

///////////////////////////////////////////////////////////////////////
// Tensor module for PySYCL
///////////////////////////////////////////////////////////////////////
PYBIND11_MODULE(tensor, m) {
  m.doc() = R"delim(
    Tensor Factories module for PySYCL
      This module provides factory functions for pysycl Tensors.
    )delim";

  tensor_base(m);
  tensor_dims(m);

  tensor_1d<double>(m);
  tensor_1d<float>(m);
  tensor_1d<int>(m);

  tensor_2d<double>(m);
  tensor_2d<float>(m);
  tensor_2d<int>(m);
}

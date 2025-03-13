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

///////////////////////////////////////////////////////////////////////
// Declaring types for tensor
///////////////////////////////////////////////////////////////////////
using Device_T = pysycl::Device;
using Module_T = py::module_;
using Tensor_double_T = pysycl::Tensor<double>;
using Tensor_int_T = pysycl::Tensor<int>;
using Tensor_float_T = pysycl::Tensor<float>;
using Variant_T = std::variant<Tensor_double_T, Tensor_int_T, Tensor_float_T>;

namespace py = pybind11;

template<typename Module_type, typename Variant_type, size_t Index = 0>
void bind_tensor(Module_type& m) {
  if constexpr (Index < std::variant_size_v<Variant_type>) {
    using Tensor_type = std::variant_alternative_t<Index, Variant_type>;
    using Scalar_type = Tensor_type::Scalar_T;
    using Vector_type = std::vector<Scalar_type>;

    const auto id = typeid(Scalar_type).name();
    std::string name = "tensor_" + std::string(id);

    if (std::string(id) == "d") {
      py::class_<Tensor_type> tensor_object(m, name.c_str(), R"delim(
        This class creates a PySYCL tensor.
      )delim");

      tensor_object.def(
          py::init<const Device_T&, const Vector_type&>(),
          R"delim(
      Default Constructor
        Constructor that creates a 1D PySYCL tensor.

        Parameters
          device: pysycl.device.device_instance
            The PySYCL device instance
          dims: List[]
            The elements of the one dimensional tensor

        Returns
          A PySYCL tensor

        Example
          >>> import pysycl
          >>>
          >>> my_device = pysycl.device()
          >>> my_tensor = pysycl.tensor(my_device, [3.3, 8.72, 1.22, -83.8])
      )delim",
          py::arg("device"),
          py::arg("dims"));
    } else {
      py::class_<Tensor_type> tensor_object(m, name.c_str());

      tensor_object.def(
          py::init<const Device_T&, const Vector_type&>(),
          py::arg("device"),
          py::arg("dims"));
    }

    bind_tensor<Module_type, Variant_type, Index + 1>(m);
  }
}

///////////////////////////////////////////////////////////////////////
// Tensor module for PySYCL
///////////////////////////////////////////////////////////////////////
PYBIND11_MODULE(tensor_types, m) {
  m.doc() = R"delim(
    Tensor module for PySYCL
      This module provides classes and functions for pysycl Tensors.
    )delim";

  bind_tensor<Module_T, Variant_T>(m);
}
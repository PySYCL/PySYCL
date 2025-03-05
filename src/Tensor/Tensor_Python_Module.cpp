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
#include <vector>

///////////////////////////////////////////////////////////////////////
// Declaring types for tensor
///////////////////////////////////////////////////////////////////////
using Device_T = pysycl::Device;
using Scalar_T = double;
using Tensor_T = pysycl::Tensor<Scalar_T>;
using Vector_T = std::vector<Scalar_T>;

namespace py = pybind11;

///////////////////////////////////////////////////////////////////////
// Tensor module for PySYCL
///////////////////////////////////////////////////////////////////////
PYBIND11_MODULE(tensor, m) {
  m.doc() = R"delim(
    Tensor module for PySYCL
      This module provides classes and functions for pysycl Tensors.
    )delim";

  py::class_<Tensor_T> tensor_object(m, "tensor", R"delim(
    This class creates a PySYCL tensor.
  )delim");

  tensor_object.def(py::init<const Device_T&, const Vector_T&>(), R"delim(
  Default Constructor
    Constructor that creates a PySYCL tensor.

    Parameters
      device: pysycl.device.device_instance
        The PySYCL device instance
      dims: List[]
        The list of the tensor dimensions

    Returns
      A PySYCL tensor

    Example
      >>> import pysycl
      >>>
      >>> my_device = pysycl.device()
      >>> my_tensor = pysycl.tensor(my_device, [3, 8])
  )delim",
  py::arg("device"),
  py::arg("dims"));
}

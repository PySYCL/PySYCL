#ifndef MATRIX_FACTORIES_PYTHON_MODULE_H
#define MATRIX_FACTORIES_PYTHON_MODULE_H

///////////////////////////////////////////////////////////////////////
// This file is part of the PySYCL software for SYCL development in
// Python.  It is licensed under the MIT licence.  A copy of
// this license, in a file named LICENSE.md, should have been
// distributed with this file.  A copy of this license is also
// currently available at "http://opensource.org/licenses/MIT".
//
// Unless explicitly stated, all contributions intentionally submitted
// to this project shall also be under the terms and conditions of this
// license, without any additional terms or conditions.
///////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////
/// \file
/// \brief Python module for an matrix factories in PySYCL.
///////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////
/// pybind11
///////////////////////////////////////////////////////////////////////
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

///////////////////////////////////////////////////////////////////////
/// local
///////////////////////////////////////////////////////////////////////
#include "../Data_Types/Data_Types.h"
#include "../Device/Device_Instance.h"
#include "Matrix_Factories.h"

namespace py = pybind11;

using Device_T = pysycl::Device_Instance;
using Data_T = pysycl::Data_Types;

///////////////////////////////////////////////////////////////////////
// Matrix Factories function
///////////////////////////////////////////////////////////////////////
void matrix_factories_module(py::module &m) {
  m.def(
      "matrix",
      [](std::tuple<int, int> dims, Device_T &device, Data_T &dtype) {
        return pysycl::matrix_factories(dims, device, dtype);
      },
      py::arg("dims"), py::arg("device"), py::arg("dtype"));
}

///////////////////////////////////////////////////////////////////////
// Matrix Factories function with input numpt array
///////////////////////////////////////////////////////////////////////
template <typename Scalar_T> void matrix_factories_numpy_module(py::module &m) {
  m.def(
      "matrix",
      [](py::array_t<Scalar_T> np_array, Device_T &device, Data_T &dtype) {
        return pysycl::matrix_factories(np_array, device, dtype);
      },
      py::arg("np_array"), py::arg("device"), py::arg("dtype"));
}

#endif // MATRIX_FACTORIES_PYTHON_MODULE_H
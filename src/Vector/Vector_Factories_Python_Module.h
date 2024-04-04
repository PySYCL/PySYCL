#ifndef VECTOR_FACTORIES_PYTHON_MODULE_H
#define VECTOR_FACTORIES_PYTHON_MODULE_H

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
/// \brief Python module for a vector object in PySYCL.
///////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////
/// pybind11
///////////////////////////////////////////////////////////////////////
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

///////////////////////////////////////////////////////////////////////
/// local
///////////////////////////////////////////////////////////////////////
#include "../Data_Types/Data_Types.h"
#include "../Device/Device_Instance.h"
#include "Vector_Factories.h"

namespace py = pybind11;

using Device_T = pysycl::Device_Instance;
using Data_T = pysycl::Data_Types;

///////////////////////////////////////////////////////////////////////
// Vector Factories function
///////////////////////////////////////////////////////////////////////
void vector_factories_module(py::module &m) {
  m.def(
      "vector",
      [](int dims, Device_T &device, Data_T &dtype) {
        return pysycl::vector_factories(dims, device, dtype);
      },
      py::arg("dims"), py::arg("device"), py::arg("dtype"));
}

///////////////////////////////////////////////////////////////////////
// Vector Factories function with input numpt array
///////////////////////////////////////////////////////////////////////
template <typename Scalar_T> void vector_factories_numpy_module(py::module &m) {
  m.def(
      "vector",
      [](py::array_t<Scalar_T> np_array, Device_T &device, Data_T &dtype) {
        return pysycl::vector_factories(np_array, device, dtype);
      },
      py::arg("np_array"), py::arg("device"), py::arg("dtype"));
}

#endif // VECTOR_FACTORIES_PYTHON_MODULE_H
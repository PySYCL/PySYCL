#ifndef VECTOR_TYPE_PYTHON_MODULE_H
#define VECTOR_TYPE_PYTHON_MODULE_H

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
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

///////////////////////////////////////////////////////////////////////
/// local
///////////////////////////////////////////////////////////////////////
#include "../Device/Device_Instance.h"
#include "Vector_Factories.h"
#include "Vector_Type.h"

namespace py = pybind11;

using Device_T = pysycl::Device_Instance;

///////////////////////////////////////////////////////////////////////
// Vector Type Double class and functions
///////////////////////////////////////////////////////////////////////
void vector_type_double_module(py::module &m) {
  using Scalar_T = double;
  using Vector_T = pysycl::Vector<Scalar_T>;
  py::class_<Vector_T>(m, "vector_type_double", R"delim(
    Description
      This class creates a PySYCL type double vector.
    )delim")
      .def(py::init<int, Device_T &>(), R"delim(
      Default Constructor
        Constructor that creates a PySYCL vector.

      Parameters
        size: int
          Number of elements.
        device:
          Optional: The target PySYCL device.

        Returns
          A PySYCL vector of type double.

        Example
          >>> import pysycl
          >>> N = 25
          >>> A_pysycl = pysycl.vector(N, device= self.device, dtype= pysycl.double)
      )delim",
           py::arg("size"), py::arg("device"))
      .def(py::init<py::array_t<Scalar_T>, pysycl::Device_Instance &>(),
           R"delim(
      NumPy Constructor
        Constructor that creates a PySYCL vector from a 1D NumPy array.

      Parameters
        np_arr: numpy.array()
          numpy array.
        device:
          Optional: The target PySYCL device.

        Returns
          A PySYCL vector.

        Example
          >>> import pysycl
          >>> import numpy as np
          >>> A_np = np.random.rand(N).astype(np.float64)
          >>> A_pysycl = pysycl.vector(A_np, device= self.device, dtype= pysycl.double)
      )delim",
           py::arg("np_arr"), py::arg("device"))
      .def("get_size", &Vector_T::get_size, R"delim(
      Description
        This function returns the number of elements.

      Returns
        The number of elements.

      Example
        >>> size = A_pysycl.get_size()
        >>> print(size)
        25
      )delim")
      .def("fill", &Vector_T::fill, R"delim(
      Description
        This function fills the vector with a constant value.

      Parameters
        C : float
          Some scalar constant.

      Example
        >>> A_pysycl.fill(12.79)
        >>> A_pysycl.mem_to_cpu()
        >>> print(A_pysycl[10])
        12.79
      )delim",
           py::arg("C"))
      .def("mem_to_gpu", &Vector_T::mem_to_gpu, R"delim(
      Description
        This function copies vector memory from CPU to GPU.

      Example
        >>> A_pysycl.mem_to_gpu()
      )delim")
      .def("mem_to_cpu", &Vector_T::mem_to_cpu, R"delim(
      Description
        This function copies array memory from GPU to CPU.

      Example
        >>> A_pysycl.mem_to_cpu()
      )delim")
      .def("max", &Vector_T::max, R"delim(
      Description
        This function finds the maximum value in the vector.

      Returns
        The maximum value.

      Example
        >>> max = A_pysycl.max()
      )delim")
      .def("min", &Vector_T::min, R"delim(
      Description
        This function finds the minimum value in the vector.

      Returns
        The minimum element value.

      Example
        >>> min = A_pysycl.min()
      )delim")
      .def("sum", &Vector_T::sum, R"delim(
      Description
        This function finds the sum of all element values in the vector.

      Returns
        The sum of all element values.

      Example
        >>> sum = A_pysycl.sum()
      )delim")
      .def("__getitem__", [](Vector_T &self, int i) { return self(i); })
      .def("__setitem__",
           [](Vector_T &self, int i, Scalar_T val) { self(i) = val; })
      .def("__add__",
           [](const Vector_T &a, const Vector_T &b) -> Vector_T {
             return a + b;
           })
      .def("__iadd__",
           [](const Vector_T &a, const Vector_T &b) { return a + b; })
      .def("__sub__",
           [](const Vector_T &a, const Vector_T &b) -> Vector_T {
             return a - b;
           })
      .def("__isub__",
           [](const Vector_T &a, const Vector_T &b) { return a - b; })
      .def("__mul__",
           [](const Vector_T &a, const Vector_T &b) -> Vector_T {
             return a * b;
           })
      .def("__imul__",
           [](const Vector_T &a, const Vector_T &b) { return a * b; })
      .def("__truediv__",
           [](const Vector_T &a, const Vector_T &b) -> Vector_T {
             return a / b;
           })
      .def("__itruediv__",
           [](const Vector_T &a, const Vector_T &b) { return a / b; });
}

///////////////////////////////////////////////////////////////////////
// Vector Type Float class and functions
///////////////////////////////////////////////////////////////////////
void vector_type_float_module(py::module &m) {
  using Scalar_T = float;
  using Vector_T = pysycl::Vector<Scalar_T>;
  py::class_<Vector_T>(m, "vector_type_float", R"delim(
    Description
      This class creates a PySYCL type float vector.
    )delim")
      .def(py::init<int, Device_T &>(),
           py::arg("size"), py::arg("device"))
      .def(py::init<py::array_t<Scalar_T>, pysycl::Device_Instance &>(),
           py::arg("np_arr"), py::arg("device"))
      .def("get_size", &Vector_T::get_size)
      .def("fill", &Vector_T::fill,
           py::arg("C"))
      .def("mem_to_gpu", &Vector_T::mem_to_gpu)
      .def("mem_to_cpu", &Vector_T::mem_to_cpu)
      .def("max", &Vector_T::max)
      .def("min", &Vector_T::min)
      .def("sum", &Vector_T::sum)
      .def("__getitem__", [](Vector_T &self, int i) { return self(i); })
      .def("__setitem__",
           [](Vector_T &self, int i, Scalar_T val) { self(i) = val; })
      .def("__add__",
           [](const Vector_T &a, const Vector_T &b) -> Vector_T {
             return a + b;
           })
      .def("__iadd__",
           [](const Vector_T &a, const Vector_T &b) { return a + b; })
      .def("__sub__",
           [](const Vector_T &a, const Vector_T &b) -> Vector_T {
             return a - b;
           })
      .def("__isub__",
           [](const Vector_T &a, const Vector_T &b) { return a - b; })
      .def("__mul__",
           [](const Vector_T &a, const Vector_T &b) -> Vector_T {
             return a * b;
           })
      .def("__imul__",
           [](const Vector_T &a, const Vector_T &b) { return a * b; })
      .def("__truediv__",
           [](const Vector_T &a, const Vector_T &b) -> Vector_T {
             return a / b;
           })
      .def("__itruediv__",
           [](const Vector_T &a, const Vector_T &b) { return a / b; });
}

///////////////////////////////////////////////////////////////////////
// Vector Type Int class and functions
///////////////////////////////////////////////////////////////////////
void vector_type_int_module(py::module &m) {
  using Scalar_T = int;
  using Vector_T = pysycl::Vector<Scalar_T>;
  py::class_<Vector_T>(m, "vector_type_int", R"delim(
    Description
      This class creates a PySYCL type int vector.
    )delim")
      .def(py::init<int, Device_T &>(),
           py::arg("size"), py::arg("device"))
      .def(py::init<py::array_t<Scalar_T>, pysycl::Device_Instance &>(),
           py::arg("np_arr"), py::arg("device"))
      .def("get_size", &Vector_T::get_size)
      .def("fill", &Vector_T::fill,
           py::arg("C"))
      .def("mem_to_gpu", &Vector_T::mem_to_gpu)
      .def("mem_to_cpu", &Vector_T::mem_to_cpu)
      .def("max", &Vector_T::max)
      .def("min", &Vector_T::min)
      .def("sum", &Vector_T::sum)
      .def("__getitem__", [](Vector_T &self, int i) { return self(i); })
      .def("__setitem__",
           [](Vector_T &self, int i, Scalar_T val) { self(i) = val; })
      .def("__add__",
           [](const Vector_T &a, const Vector_T &b) -> Vector_T {
             return a + b;
           })
      .def("__iadd__",
           [](const Vector_T &a, const Vector_T &b) { return a + b; })
      .def("__sub__",
           [](const Vector_T &a, const Vector_T &b) -> Vector_T {
             return a - b;
           })
      .def("__isub__",
           [](const Vector_T &a, const Vector_T &b) { return a - b; })
      .def("__mul__",
           [](const Vector_T &a, const Vector_T &b) -> Vector_T {
             return a * b;
           })
      .def("__imul__",
           [](const Vector_T &a, const Vector_T &b) { return a * b; })
      .def("__truediv__",
           [](const Vector_T &a, const Vector_T &b) -> Vector_T {
             return a / b;
           })
      .def("__itruediv__",
           [](const Vector_T &a, const Vector_T &b) { return a / b; });
}

#endif // VECTOR_TYPE_PYTHON_MODULE_H

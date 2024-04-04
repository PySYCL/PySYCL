#ifndef MATRIX_TYPE_PYTHON_MODULE_H
#define MATRIX_TYPE_PYTHON_MODULE_H

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
/// \brief Python module for an matrix object in PySYCL.
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
#include "Matrix_Type.h"

namespace py = pybind11;

///////////////////////////////////////////////////////////////////////
// Device class and functions
///////////////////////////////////////////////////////////////////////
void matrix_type_double_module(py::module &m) {
  using Scalar_T = double;
  using Matrix_T = pysycl::Matrix<Scalar_T>;
  py::class_<Matrix_T>(m, "matrix_type_double", R"delim(
    Description
      This class creates a PySYCL type double matrix.
    )delim")
      .def(py::init<int, int, pysycl::Device_Instance &>(), R"delim(
      Default Constructor
        Constructor that creates a 2D PySYCL array.

      Parameters
        cols: int
          Number of columns.
        rows: int
          Number of rows.
        device:
          Optional: The target PySYCL device.

        Returns
          A PySYCL matrix of type double.

        Example
          >>> import pysycl
          >>> M = 10
          >>> N = 12
          >>> A = pysycl.matrix((M, N), device= self.device, dtype= pysycl.double)
      )delim",
           py::arg("rows"), py::arg("cols"), py::arg("device"))
      .def(py::init<py::array_t<Scalar_T>, pysycl::Device_Instance &>(),
           R"delim(
      NumPy Constructor
        Constructor that creates a PySYCL matrix from a 2D NumPy array.

      Parameters
        np_arr: numpy.array()
          numpy array.
        device:
          Optional: The target PySYCL device.

        Returns
          A PySYCL matrix.

        Example
          >>> import pysycl
          >>> import numpy as np
          >>> A_np = np.random.rand(M, N).astype(np.float64)
          >>> A_pysycl = pysycl.matrix(A_np, device= self.device, dtype= pysycl.double)
      )delim",
           py::arg("np_arr"), py::arg("device"))
      .def("num_rows", &Matrix_T::num_rows, R"delim(
      Description
        This function returns the number of rows.

      Returns
        The number of rows.

      Example
        >>> rows = A_pysycl.num_rows()
        >>> print(rows)
        12
      )delim")
      .def("num_cols", &Matrix_T::num_cols, R"delim(
      Description
        This function returns the number of columns.

      Returns
        The number of columns.

      Example
        >>> rows = A_pysycl.num_cols()
        >>> print(cols)
        10
      )delim")
      .def("fill", &Matrix_T::fill, R"delim(
      Description
        This function fills the matrix with a constant value.

      Parameters
        C : Scalar_T
          Some scalar constant.

      Example
        >>> A_pysycl.fill(45.0)
        >>> A_pysycl.mem_to_cpu()
        >>> print(A_pysycl[9, 7])
        45.0
      )delim",
           py::arg("C"))
      .def("mem_to_gpu", &Matrix_T::mem_to_gpu, R"delim(
      Description
        This function copies array memory from CPU to GPU.

      Example
        >>> A.mem_to_gpu()
      )delim")
      .def("mem_to_cpu", &Matrix_T::mem_to_cpu, R"delim(
      Description
        This function copies array memory from GPU to CPU.

      Example
        >>> A_pysycl.mem_to_cpu()
      )delim")
      .def("max", &Matrix_T::max, R"delim(
      Description
        This function finds the maximum value in the matrix.

      Returns
        The maximum value.

      Example
        >>> max = A_pysycl.max()
      )delim")
      .def("min", &Matrix_T::min, R"delim(
      Description
        This function finds the minimum value in the matrix.

      Returns
        The minimum element value.

      Example
        >>> min = A_pysycl.min()
      )delim")
      .def("sum", &Matrix_T::sum, R"delim(
      Description
        This function finds the sum of all element values in the matrix.

      Returns
        The sum of all element values.

      Example
        >>> sum = A.sum()
      )delim")
      .def("transpose", &Matrix_T::transpose, R"delim(
      Description
        This function finds the tranpose of the matrix.

      Example
        >>> A_pysycl.transpose()
      )delim")
      .def("__getitem__",
           [](Matrix_T &self, std::pair<int, int> idx) {
             return self(idx.first, idx.second);
           })
      .def("__setitem__",
           [](Matrix_T &self, std::pair<int, int> idx, Scalar_T val) {
             self(idx.first, idx.second) = val;
           })
      .def("__add__",
           [](Matrix_T &a, Matrix_T &b) -> Matrix_T { return a + b; })
      .def("__iadd__", [](Matrix_T &a, Matrix_T &b) { return a + b; })
      .def("__sub__",
           [](Matrix_T &a, Matrix_T &b) -> Matrix_T { return a - b; })
      .def("__isub__", [](Matrix_T &a, Matrix_T &b) { return a - b; })
      .def("__mul__",
           [](Matrix_T &a, Matrix_T &b) -> Matrix_T { return a * b; })
      .def("__imul__", [](Matrix_T &a, Matrix_T &b) { return a * b; })
      .def("__truediv__",
           [](Matrix_T &a, Matrix_T &b) -> Matrix_T { return a / b; })
      .def("__itruediv__", [](Matrix_T &a, Matrix_T &b) { return a / b; });
}

void matrix_type_float_module(py::module &m) {
  using Scalar_T = float;
  using Matrix_T = pysycl::Matrix<Scalar_T>;
  py::class_<Matrix_T>(m, "matrix_type_float", R"delim(
    Description
      This class creates a PySYCL type float matrix.
    )delim")
      .def(py::init<int, int, pysycl::Device_Instance &>(),
           py::arg("rows"), py::arg("cols"), py::arg("device"))
      .def(py::init<py::array_t<Scalar_T>, pysycl::Device_Instance &>(),
           py::arg("np_arr"), py::arg("device"))
      .def("num_rows", &Matrix_T::num_rows)
      .def("num_cols", &Matrix_T::num_cols)
      .def("fill", &Matrix_T::fill,
           py::arg("C"))
      .def("mem_to_gpu", &Matrix_T::mem_to_gpu)
      .def("mem_to_cpu", &Matrix_T::mem_to_cpu )
      .def("max", &Matrix_T::max)
      .def("min", &Matrix_T::min)
      .def("sum", &Matrix_T::sum)
      .def("transpose", &Matrix_T::transpose)
      .def("__getitem__",
           [](Matrix_T &self, std::pair<int, int> idx) {
             return self(idx.first, idx.second);
           })
      .def("__setitem__",
           [](Matrix_T &self, std::pair<int, int> idx, Scalar_T val) {
             self(idx.first, idx.second) = val;
           })
      .def("__add__",
           [](Matrix_T &a, Matrix_T &b) -> Matrix_T { return a + b; })
      .def("__iadd__", [](Matrix_T &a, Matrix_T &b) { return a + b; })
      .def("__sub__",
           [](Matrix_T &a, Matrix_T &b) -> Matrix_T { return a - b; })
      .def("__isub__", [](Matrix_T &a, Matrix_T &b) { return a - b; })
      .def("__mul__",
           [](Matrix_T &a, Matrix_T &b) -> Matrix_T { return a * b; })
      .def("__imul__", [](Matrix_T &a, Matrix_T &b) { return a * b; })
      .def("__truediv__",
           [](Matrix_T &a, Matrix_T &b) -> Matrix_T { return a / b; })
      .def("__itruediv__", [](Matrix_T &a, Matrix_T &b) { return a / b; });
}

void matrix_type_int_module(py::module &m) {
  using Scalar_T = int;
  using Matrix_T = pysycl::Matrix<Scalar_T>;
  py::class_<Matrix_T>(m, "matrix_type_int", R"delim(
    Description
      This class creates a PySYCL type float matrix.
    )delim")
      .def(py::init<int, int, pysycl::Device_Instance &>(),
           py::arg("rows"), py::arg("cols"), py::arg("device"))
      .def(py::init<py::array_t<Scalar_T>, pysycl::Device_Instance &>(),
           py::arg("np_arr"), py::arg("device"))
      .def("num_rows", &Matrix_T::num_rows)
      .def("num_cols", &Matrix_T::num_cols)
      .def("fill", &Matrix_T::fill,
           py::arg("C"))
      .def("mem_to_gpu", &Matrix_T::mem_to_gpu)
      .def("mem_to_cpu", &Matrix_T::mem_to_cpu )
      .def("max", &Matrix_T::max)
      .def("min", &Matrix_T::min)
      .def("sum", &Matrix_T::sum)
      .def("transpose", &Matrix_T::transpose)
      .def("__getitem__",
           [](Matrix_T &self, std::pair<int, int> idx) {
             return self(idx.first, idx.second);
           })
      .def("__setitem__",
           [](Matrix_T &self, std::pair<int, int> idx, Scalar_T val) {
             self(idx.first, idx.second) = val;
           })
      .def("__add__",
           [](Matrix_T &a, Matrix_T &b) -> Matrix_T { return a + b; })
      .def("__iadd__", [](Matrix_T &a, Matrix_T &b) { return a + b; })
      .def("__sub__",
           [](Matrix_T &a, Matrix_T &b) -> Matrix_T { return a - b; })
      .def("__isub__", [](Matrix_T &a, Matrix_T &b) { return a - b; })
      .def("__mul__",
           [](Matrix_T &a, Matrix_T &b) -> Matrix_T { return a * b; })
      .def("__imul__", [](Matrix_T &a, Matrix_T &b) { return a * b; })
      .def("__truediv__",
           [](Matrix_T &a, Matrix_T &b) -> Matrix_T { return a / b; })
      .def("__itruediv__", [](Matrix_T &a, Matrix_T &b) { return a / b; });
}

#endif // MATRIX_TYPE_PYTHON_MODULE_H
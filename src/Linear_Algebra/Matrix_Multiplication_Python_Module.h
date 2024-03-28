#ifndef MATRIX_MULTIPLICATION_PYTHON_MODULE_H
#define MATRIX_MULTIPLICATION_PYTHON_MODULE_H

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
/// \brief Python module for a matrix multiplication in PySYCL.
///////////////////////////////////////////////////////////////////////

/// Linear Algebra functionalities in PySYCL

///////////////////////////////////////////////////////////////////////
// pybind11
///////////////////////////////////////////////////////////////////////
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

///////////////////////////////////////////////////////////////////////
// local
///////////////////////////////////////////////////////////////////////
#include "../Matrix/Matrix_Type.h"
#include "Matrix_Multiplication.h"

namespace py = pybind11;

///////////////////////////////////////////////////////////////////////
// Matrix multiplication function
///////////////////////////////////////////////////////////////////////
template<typename Scalar_T>
void matmul_module_double(py::module &m) {
  using Matrix_T = pysycl::Matrix<Scalar_T>;
  m.def("matmul", &pysycl::matmul<Matrix_T>, R"delim(
    Description
      This function evaluates a matrix multiplication and returns the result.

    Parameters
      A : pysycl.matrix
        The first matrix for multiplication.

      B : pysycl.matrix
        The second matrix for multiplication.

      wg_size : int
        Optional: Work group size

    Example
      >>> import pysycl
      >>>
      >>> M = 4000
      >>> N = 800
      >>> P = 2500
      >>>
      >>> device = pysycl.device.get_device(0,0)
      >>>
      >>> A = pysycl.matrix((M, N), device= device, dtype= pysycl.double)
      >>> B = pysycl.matrix((N, P), device= device, dtype= pysycl.double)
      >>>
      >>> C = pysycl.matrix((N, P), device= device, dtype= pysycl.double)
      >>>
      >>> A.fill(8.0)
      >>> B.fill(3.0)
      >>> pysycl.linalg.matmul(A, B, C)
      >>>
      >>> C.mem_to_cpu()
      >>> print(C[30, 50])
      19200.0
  )delim",
        py::arg("A"), py::arg("B"), py::arg("C"), py::arg("wg_size"));
}

///////////////////////////////////////////////////////////////////////
// Matrix multiplication function float
///////////////////////////////////////////////////////////////////////
template<typename Scalar_T>
void matmul_module_float(py::module &m) {
  using Matrix_T = pysycl::Matrix<Scalar_T>;
  m.def("matmul", &pysycl::matmul<Matrix_T>,
        py::arg("A"), py::arg("B"), py::arg("C"), py::arg("wg_size"));
}

///////////////////////////////////////////////////////////////////////
// Matrix multiplication function int
///////////////////////////////////////////////////////////////////////
template<typename Scalar_T>
void matmul_module_int(py::module &m) {
  using Matrix_T = pysycl::Matrix<Scalar_T>;
  m.def("matmul", &pysycl::matmul<Matrix_T>,
        py::arg("A"), py::arg("B"), py::arg("C"), py::arg("wg_size"));
}

///////////////////////////////////////////////////////////////////////
// Binding all scalar variants of the matmul function
///////////////////////////////////////////////////////////////////////
void matmul_module(py::module &m) {
  matmul_module_double<double>(m);
  matmul_module_float<float>(m);
  matmul_module_int<int>(m);
}

#endif // MATRIX_MULTIPLICATION_PYTHON_MODULE_H
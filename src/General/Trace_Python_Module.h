#ifndef TRACE_PYTHON_MODULE_H
#define TRACE_PYTHON_MODULE_H

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
/// \brief Python module for to get the trace of a matrix in PySYCL.
///////////////////////////////////////////////////////////////////////

/// General functionalities in PySYCL

///////////////////////////////////////////////////////////////////////
// pybind11
///////////////////////////////////////////////////////////////////////
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

///////////////////////////////////////////////////////////////////////
// local
///////////////////////////////////////////////////////////////////////
#include "../Matrix/Matrix_Type.h"
#include "Trace.h"

namespace py = pybind11;

///////////////////////////////////////////////////////////////////////
// Trace function double
///////////////////////////////////////////////////////////////////////
template <typename Scalar_T> void trace_module_double(py::module &m) {
  using Matrix_T = pysycl::Matrix<Scalar_T>;
  m.def("trace", &pysycl::trace<Matrix_T>, R"delim(
    Description
      This function evaluates the trace of a matrix and returns the result.

    Parameters
      A : pysycl.matrix
        The input matrix.

    Example
      >>> import pysycl
      >>>
      >>> N = 10
      >>> device = pysycl.device.get_device(0,0)
      >>>
      >>> A = pysycl.matrix((N, N), device= device, dtype= pysycl.float)
      >>> A.fill(8.0)
      >>> A = pysycl.trace(A)
      >>>
      >>> print(trace)
      80.0
  )delim",
        py::arg("A"));
}

///////////////////////////////////////////////////////////////////////
// Trace function float
///////////////////////////////////////////////////////////////////////
template <typename Scalar_T> void trace_module_float(py::module &m) {
  using Matrix_T = pysycl::Matrix<Scalar_T>;
  m.def("trace", &pysycl::trace<Matrix_T>, py::arg("A"));
}

///////////////////////////////////////////////////////////////////////
// Trace function int
///////////////////////////////////////////////////////////////////////
template <typename Scalar_T> void trace_module_int(py::module &m) {
  using Matrix_T = pysycl::Matrix<Scalar_T>;
  m.def("trace", &pysycl::trace<Matrix_T>, py::arg("A"));
}

///////////////////////////////////////////////////////////////////////
// Binding all scalar variants of the trace function
///////////////////////////////////////////////////////////////////////
void trace_module(py::module &m) {
  trace_module_double<double>(m);
  trace_module_float<float>(m);
  trace_module_int<int>(m);
}

#endif // TRACE_PYTHON_MODULE_H
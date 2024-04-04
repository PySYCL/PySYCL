#ifndef FFT_1D_PYTHON_MODULE_H
#define FFT_1D_PYTHON_MODULE_H

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
/// \brief Python module for a 1-dimensional fft in PySYCL.
///////////////////////////////////////////////////////////////////////

/// FFT Functions in PySYCL

///////////////////////////////////////////////////////////////////////
// pybind11
///////////////////////////////////////////////////////////////////////
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

///////////////////////////////////////////////////////////////////////
// local
///////////////////////////////////////////////////////////////////////
#include "../Vector/Vector_Type.h"
#include "FFT_1D.h"

namespace py = pybind11;

///////////////////////////////////////////////////////////////////////
// FFT 1D function double
///////////////////////////////////////////////////////////////////////
void fft1d_module_double(py::module &m) {
  using Vector_T = pysycl::Vector<double>;
  m.def("fft1d", &pysycl::fft1d<Vector_T>, R"delim(
    Description
      This function evaluates the 1-dimensional fft on a pysycl vector.

    Parameters
      A : pysycl.vector
        The vector to transform.

    Example
      >>> import pysycl
      >>> A_np = np.random.rand(N).astype(np.float64)
      >>> A_pysycl = pysycl.vector(A_np, device = device, dtype= pysycl.double)
      >>> fft = pysycl.fft1d(A_pysycl)
  )delim",
        py::arg("A"));
}

///////////////////////////////////////////////////////////////////////
// FFT 1D function float
///////////////////////////////////////////////////////////////////////
void fft1d_module_float(py::module &m) {
  using Vector_T = pysycl::Vector<float>;
  m.def("fft1d", &pysycl::fft1d<Vector_T>,
        py::arg("A"));
}

///////////////////////////////////////////////////////////////////////
// Binding all scalar variants of the fft1d function
///////////////////////////////////////////////////////////////////////
void fft1d_module(py::module &m) {
  fft1d_module_double(m);
  fft1d_module_float(m);
}

#endif // FFT_1D_PYTHON_MODULE_H
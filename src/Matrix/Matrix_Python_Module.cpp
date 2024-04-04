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
/// \brief Python module for matrices in PySYCL.
///////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////
/// pybind11
///////////////////////////////////////////////////////////////////////
#include <pybind11/pybind11.h>

///////////////////////////////////////////////////////////////////////
/// local
///////////////////////////////////////////////////////////////////////
#include "Matrix_Factories_Python_Module.h"
#include "Matrix_Type_Python_Module.h"

namespace py = pybind11;

///////////////////////////////////////////////////////////////////////
// Vector module for PySYCL
///////////////////////////////////////////////////////////////////////
PYBIND11_MODULE(matrix, m) {
  m.doc() = R"delim(
    Matrix module for PySYCL
      This module provides classes and functions for creating PySYCL matrices.
    )delim";

  matrix_type_double_module(m);
  matrix_type_float_module(m);
  matrix_type_int_module(m);
  matrix_factories_module(m);
  matrix_factories_numpy_module<double>(m);
  matrix_factories_numpy_module<float>(m);
  matrix_factories_numpy_module<int>(m);
}

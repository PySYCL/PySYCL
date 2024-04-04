#ifndef DATA_TYPES_PYTHON_MODULE_H
#define DATA_TYPES_PYTHON_MODULE_H

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
/// \brief Python module for data types in PySYCL.
///////////////////////////////////////////////////////////////////////

/// Data Types in PySYCL

///////////////////////////////////////////////////////////////////////
// pybind11
///////////////////////////////////////////////////////////////////////
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

///////////////////////////////////////////////////////////////////////
// local
///////////////////////////////////////////////////////////////////////
#include "Data_Types.h"

namespace py = pybind11;

using Data_T = pysycl::Data_Types;

void data_types_module(py::module &m) {
  py::enum_<Data_T>(m, "data_types")
      .value("double", Data_T::DOUBLE)
      .value("float", Data_T::FLOAT)
      .value("int", Data_T::INT)
      .export_values();
}

#endif // DATA_TYPES_PYTHON_MODULE_H

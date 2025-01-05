#ifndef DEVICE_INQUIRY_PYTHON_MODULE_H
#define DEVICE_INQUIRY_PYTHON_MODULE_H

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
/// \brief Python module for device inquiry in PySYCL.
///////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////
// local
///////////////////////////////////////////////////////////////////////
#include "Device_Inquiry.h"

///////////////////////////////////////////////////////////////////////
// pybind11
///////////////////////////////////////////////////////////////////////
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

///////////////////////////////////////////////////////////////////////
// stl
///////////////////////////////////////////////////////////////////////
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////
// Defining types
///////////////////////////////////////////////////////////////////////
using Vector_T = std::vector<std::vector<int>>;

namespace py = pybind11;

///////////////////////////////////////////////////////////////////////
// Device inquiry function
///////////////////////////////////////////////////////////////////////
void device_inquiry_module(py::module &m) {
  m.def("get_device_list", &pysycl::get_device_list<Vector_T>, R"delim(
    Description
      This function returns a list of all available devices.

    Returns
      A list of available PySYCL devices.

    Example
      >>> import pysycl
      >>> my_devices = device.get_device_list()
      >>> print(my_devices)
      [[0, 0]]
  )delim");

  m.def("output_device_list", &pysycl::output_device_list, R"delim(
    Description
      This function outputs a list of all available devices.

    Returns
      None.

    Example
      >>> import pysycl
      >>> pysycl.device.output_device_list()
      ['NVIDIA GeForce RTX 3060 Laptop GPU [0, 0]']
  )delim");
}

#endif // DEVICE_INQUIRY_PYTHON_MODULE_H
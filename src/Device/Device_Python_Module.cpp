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
/// \brief Python module for device in PySYCL.
///////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////
// local
///////////////////////////////////////////////////////////////////////
#include "Device_Inquiry_Python_Module.h"
#include "Device_Instance_Python_Module.h"

///////////////////////////////////////////////////////////////////////
// pybind11
///////////////////////////////////////////////////////////////////////
#include <pybind11/pybind11.h>

namespace py = pybind11;

///////////////////////////////////////////////////////////////////////
// Device module for PySYCL
///////////////////////////////////////////////////////////////////////

PYBIND11_MODULE(device, m) {
  m.doc() = R"delim(
    Device module for PySYCL
      This module provides classes and functions for selecting SYCL devices.
    )delim";

  device_inquiry_module(m);
  device_instance_module(m);
}

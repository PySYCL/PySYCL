#ifndef DEVICE_PYTHON_MODULE_H
#define DEVICE_PYTHON_MODULE_H

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
#include "Device.h"

///////////////////////////////////////////////////////////////////////
// pybind11
///////////////////////////////////////////////////////////////////////
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

///////////////////////////////////////////////////////////////////////
// Declaring types for device
///////////////////////////////////////////////////////////////////////
using Device_T = pysycl::Device;

namespace py = pybind11;

///////////////////////////////////////////////////////////////////////
// Device class and functions
///////////////////////////////////////////////////////////////////////
PYBIND11_MODULE(device, m) {
  m.doc() = R"delim(
    Device module for PySYCL
      This module provides classes and functions for selecting pysycl devices.
    )delim";

  py::class_<Device_T> device_instance(
      m, "device_instance", R"delim(
    Description
      This class creates a PySYCL device instance.
    )delim");

  device_instance
      .def(py::init<>(), "Default constructor", R"delim(
    Description
      This is the default constructor for a device instance.

    Returns
      A PySYCL device instance.

    Example
      >>> import pysycl
      >>> my_device = pysycl.device()
    )delim")
      .def(
          py::init<const int, const int>(),
          R"delim(
    Description
      This is the a constructor for a device instance that
      takes the platform and device index as input parameters.

    Parameters
      platform_idx : int
        Platform index
      device_idx : int
        Device index

    Returns
      A PySYCL device instance.

    Example
      >>> import pysycl
      >>> my_device = pysycl.device()
    )delim",
          py::arg("platform_idx"),
          py::arg("device_idx"))
      .def("name", &Device_T::name, R"delim(
    Description
      This function outputs the device name.

    Returns
      The PySYCL device name.

    Example
      >>> print(my_device.name())
      NVIDIA GeForce RTX 3060 Laptop GPU

    )delim")
      .def("vendor", &Device_T::vendor, R"delim(
    Description
      This function returns the device vendor.

    Returns
      The PySYCL device vendor.

    Example
      >>> print(my_device.vendor())
      NVIDIA Corporation
    )delim")
      .def(
          "__eq__",
          [](const Device_T& di1,
             const Device_T& di2) { return di1 == di2; })
      .def(
          "__ne__",
          [](const Device_T& di1,
             const Device_T& di2) { return di1 != di2; })
      .def(
          "__lt__",
          [](const Device_T& di1,
             const Device_T& di2) { return di1 < di2; })
      .def(
          "__le__",
          [](const Device_T& di1,
             const Device_T& di2) { return di1 <= di2; })
      .def(
          "__gt__",
          [](const Device_T& di1,
             const Device_T& di2) { return di1 > di2; })
      .def(
          "__ge__",
          [](const Device_T& di1,
             const Device_T& di2) { return di1 >= di2; });
}

#endif // DEVICE_INSTANCE_PYTHON_MODULE_H
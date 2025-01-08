#ifndef DEVICE_INSTANCE_PYTHON_MODULE_H
#define DEVICE_INSTANCE_PYTHON_MODULE_H

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
/// \brief Python module for device instance in PySYCL.
///////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////
// local
///////////////////////////////////////////////////////////////////////
#include "Device_Instance.h"

///////////////////////////////////////////////////////////////////////
// pybind11
///////////////////////////////////////////////////////////////////////
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

///////////////////////////////////////////////////////////////////////
// Device class and functions
///////////////////////////////////////////////////////////////////////
void device_instance_module(py::module &m) {
  py::class_<pysycl::Device_Instance> device_instance(m, "device_instance", R"delim(
    Description
      This class creates a PySYCL device instance.
    )delim");

  device_instance.def(py::init<>(), "Default constructor", R"delim(
    Description
      This is the default constructor for a device instance.

    Returns
      A PySYCL device instance.

    Example
      >>> import pysycl
      >>> my_device = pysycl.device.device_instance()
    )delim")
  .def(py::init<const int, const int>(), R"delim(
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
      >>> my_device = pysycl.device.device_instance()
    )delim",
    py::arg("platform_idx"),
    py::arg("device_idx"))
  .def("name", &pysycl::Device_Instance::name, R"delim(
    Description
      This function outputs the device name.

    Returns
      The PySYCL device name.

    Example
      >>> print(my_device.name())
      NVIDIA GeForce RTX 3060 Laptop GPU

    )delim")
  .def("vendor", &pysycl::Device_Instance::vendor, R"delim(
    Description
      This function returns the device vendor.

    Returns
      The PySYCL device vendor.

    Example
      >>> print(my_device.vendor())
      NVIDIA Corporation
    )delim")
    .def("__eq__",
          [](const pysycl::Device_Instance &di1,
            const pysycl::Device_Instance &di2) { return di1 == di2; })
    .def("__ne__",
          [](const pysycl::Device_Instance &di1,
            const pysycl::Device_Instance &di2) { return di1 != di2; })
    .def("__lt__",
          [](const pysycl::Device_Instance &di1,
            const pysycl::Device_Instance &di2) { return di1 < di2; })
    .def("__le__",
          [](const pysycl::Device_Instance &di1,
            const pysycl::Device_Instance &di2) { return di1 <= di2; })
    .def("__gt__",
          [](const pysycl::Device_Instance &di1,
            const pysycl::Device_Instance &di2) { return di1 > di2; })
    .def("__ge__",
          [](const pysycl::Device_Instance &di1,
            const pysycl::Device_Instance &di2) { return di1 >= di2; });
}

#endif // DEVICE_INSTANCE_PYTHON_MODULE_H
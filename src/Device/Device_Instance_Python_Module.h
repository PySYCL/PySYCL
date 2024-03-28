#ifndef DEVICE_INSTANCE_PYTHON_MODULE_H
#define DEVICE_INSTANCE_PYTHON_MODULE_H

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
/// \brief Python module for device instance in PySYCL.
///////////////////////////////////////////////////////////////////////

/// Device Management in PySYCL

///////////////////////////////////////////////////////////////////////
// pybind11
///////////////////////////////////////////////////////////////////////
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

///////////////////////////////////////////////////////////////////////
// local
///////////////////////////////////////////////////////////////////////
#include "Device_Instance.h"
#include "Device_Manager.h"

namespace py = pybind11;

///////////////////////////////////////////////////////////////////////
// Device class and functions
///////////////////////////////////////////////////////////////////////
void device_instance_module(py::module &m) {
  py::class_<pysycl::Device_Instance>(m, "device_instance", R"delim(
    Description
      This class creates a PySYCL device instance.
    )delim")
      .def("name", &pysycl::Device_Instance::name, R"delim(
      Description
        This function returns the device name.

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
      .def("__eq__", [](const pysycl::Device_Instance& di1,
                        const pysycl::Device_Instance& di2){return di1 == di2;})
      .def("__ne__", [](const pysycl::Device_Instance& di1,
                        const pysycl::Device_Instance& di2){return di1 != di2;})
      .def("__lt__", [](const pysycl::Device_Instance& di1,
                        const pysycl::Device_Instance& di2){return di1 < di2;})
      .def("__le__", [](const pysycl::Device_Instance& di1,
                        const pysycl::Device_Instance& di2){return di1 <= di2;})
      .def("__gt__", [](const pysycl::Device_Instance& di1,
                        const pysycl::Device_Instance& di2){return di1 > di2;})
      .def("__ge__", [](const pysycl::Device_Instance& di1,
                        const pysycl::Device_Instance& di2){return di1 >= di2;});
}

#endif // DEVICE_INSTANCE_PYTHON_MODULE_H
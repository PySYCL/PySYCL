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
// gtest
///////////////////////////////////////////////////////////////////////
#include <gtest/gtest.h>

///////////////////////////////////////////////////////////////////////
// stl
///////////////////////////////////////////////////////////////////////
#include <iostream>
#include <sstream>
#include <vector>

///////////////////////////////////////////////////////////////////////
// sycl
///////////////////////////////////////////////////////////////////////
#include <sycl/sycl.hpp>

///////////////////////////////////////////////////////////////////////
// sycl
///////////////////////////////////////////////////////////////////////
#include "Device.h"

///////////////////////////////////////////////////////////////////////
// Defining types
///////////////////////////////////////////////////////////////////////
using Vector_T = std::vector<std::vector<int>>;

///////////////////////////////////////////////////////////////////////
// Device Test 1 (check platform and device index)
///////////////////////////////////////////////////////////////////////
TEST(Device, test1) {
  const auto& platforms = sycl::platform::get_platforms();

  for (int i = 0; i < platforms.size(); ++i) {
    const auto& devices = platforms[i].get_devices();

    for (int j = 0; j < devices.size(); ++j) {
      auto my_device = pysycl::Device(i, j);
      ASSERT_EQ(i, my_device.get_platform_index());
      ASSERT_EQ(j, my_device.get_device_index());
    }
  }
}

///////////////////////////////////////////////////////////////////////
// Device Test 2 (check device name and vendor)
///////////////////////////////////////////////////////////////////////
TEST(Device, test2) {
  const auto& platforms = sycl::platform::get_platforms();

  for (int i = 0; i < platforms.size(); ++i) {
    const auto& devices = platforms[i].get_devices();

    for (int j = 0; j < devices.size(); ++j) {
      auto my_device = pysycl::Device(i, j);
      auto Q = sycl::queue(devices[j]);
      ASSERT_EQ(
          Q.get_device().get_info<sycl::info::device::name>(),
          my_device.name());
      ASSERT_EQ(
          Q.get_device().get_info<sycl::info::device::vendor>(),
          my_device.vendor());
    }
  }
}
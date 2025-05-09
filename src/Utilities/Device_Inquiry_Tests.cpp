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
#include "Device_Inquiry.h"

///////////////////////////////////////////////////////////////////////
// Defining types
///////////////////////////////////////////////////////////////////////
using Vector_T = std::vector<std::vector<int>>;

///////////////////////////////////////////////////////////////////////
// Device Inquiry Test 1 (get device list)
///////////////////////////////////////////////////////////////////////
TEST(DeviceInquiry, test1) {
  auto my_devices = pysycl::get_device_list<Vector_T>();

  ASSERT_FALSE(my_devices.empty()) << "The device list is empty!";
}

///////////////////////////////////////////////////////////////////////
// Device Inquiry Test 2 (output device list)
///////////////////////////////////////////////////////////////////////
TEST(DeviceInquiry, test2) {
  std::ostringstream oss;
  std::streambuf* sb = std::cout.rdbuf(oss.rdbuf());

  pysycl::output_device_list();
  std::cout.rdbuf(sb);

  std::string output = oss.str();
  ASSERT_FALSE(output.empty()) << "The device list did not output!";
}

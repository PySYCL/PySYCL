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
// local
///////////////////////////////////////////////////////////////////////
#include "../Device/Device.h"
#include "Tensor.h"

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
// Defining types
///////////////////////////////////////////////////////////////////////
using Scalar_T = double;
using Device_T = pysycl::Device;
using Tensor_T = pysycl::Tensor<Scalar_T>;

///////////////////////////////////////////////////////////////////////
// Tensor Test 1
///////////////////////////////////////////////////////////////////////
TEST(Tensor, test1) {
  auto device = Device_T(0, 0);

  auto tensor0 = Tensor_T(device, {8,    10,   7});
  auto tensor1 = Tensor_T(device, {9,    1,    1,  8172});
  auto tensor2 = Tensor_T(device, {12,   12,   10, 82, 772});
  auto tensor3 = Tensor_T(device, {8002, 2110});
  auto tensor4 = Tensor_T(device, {237});

  ASSERT_EQ(8*10*7,          tensor0.len());
  ASSERT_EQ(9*1*1*8172,      tensor1.len());
  ASSERT_EQ(12*12*10*82*772, tensor2.len());
  ASSERT_EQ(8002*2110,       tensor3.len());
  ASSERT_EQ(237,             tensor4.len());
}

///////////////////////////////////////////////////////////////////////
// Tensor Test 2
///////////////////////////////////////////////////////////////////////
TEST(Tensor, test2) {
  auto device = Device_T(0, 0);

  auto tensor0 = Tensor_T(device, {8,    10,   7});
  auto tensor1 = Tensor_T(device, {9,    1,    1,  8172});
  auto tensor2 = Tensor_T(device, {12,   12,   10, 82, 772});
  auto tensor3 = Tensor_T(device, {8002, 2110});
  auto tensor4 = Tensor_T(device, {237});

  ASSERT_EQ(3, tensor0.num_dims());
  ASSERT_EQ(4, tensor1.num_dims());
  ASSERT_EQ(5, tensor2.num_dims());
  ASSERT_EQ(2, tensor3.num_dims());
  ASSERT_EQ(1, tensor4.num_dims());
}
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
#include "../Utilities/Generate_Vectors.h"
#include "Tensor.h"

///////////////////////////////////////////////////////////////////////
// pybind
///////////////////////////////////////////////////////////////////////
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>

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

namespace py = pybind11;

///////////////////////////////////////////////////////////////////////
// Tensor Test 1
///////////////////////////////////////////////////////////////////////
TEST(Tensor, test1) {
  auto device = Device_T(0, 0);

  py::scoped_interpreter guard{};

  auto tensor0 = Tensor_T(device, py::make_tuple(8, 10, 7));
  auto tensor1 = Tensor_T(device, py::make_tuple(9, 1, 1, 8172));
  auto tensor2 = Tensor_T(device, py::make_tuple(12, 12, 10, 82, 772));
  auto tensor3 = Tensor_T(device, py::make_tuple(8002, 2110));
  auto tensor4 = Tensor_T(device, py::make_tuple(237));

  ASSERT_EQ(8 * 10 * 7, tensor0.len());
  ASSERT_EQ(9 * 1 * 1 * 8172, tensor1.len());
  ASSERT_EQ(12 * 12 * 10 * 82 * 772, tensor2.len());
  ASSERT_EQ(8002 * 2110, tensor3.len());
  ASSERT_EQ(237, tensor4.len());
}

///////////////////////////////////////////////////////////////////////
// Tensor Test 2
///////////////////////////////////////////////////////////////////////
TEST(Tensor, test2) {
  auto device = Device_T(0, 0);

  py::scoped_interpreter guard{};

  auto tensor0 = Tensor_T(device, py::make_tuple(8, 10, 7));
  auto tensor1 = Tensor_T(device, py::make_tuple(9, 1, 1, 8172));
  auto tensor2 = Tensor_T(device, py::make_tuple(12, 12, 10, 82, 772));
  auto tensor3 = Tensor_T(device, py::make_tuple(8002, 2110));
  auto tensor4 = Tensor_T(device, py::make_tuple(237));

  ASSERT_EQ(3, tensor0.num_dims());
  ASSERT_EQ(4, tensor1.num_dims());
  ASSERT_EQ(5, tensor2.num_dims());
  ASSERT_EQ(2, tensor3.num_dims());
  ASSERT_EQ(1, tensor4.num_dims());
}

///////////////////////////////////////////////////////////////////////
// Tensor Test 3
///////////////////////////////////////////////////////////////////////
TEST(Tensor, test3) {
  auto device = Device_T(0, 0);

  std::vector<Scalar_T> vec1D = {1.853, -2.22, 0.213, 512.66};
  auto tensor1D = Tensor_T(device, vec1D);

  ASSERT_EQ(4, tensor1D.len());
  ASSERT_EQ(1, tensor1D.num_dims());

  for (int i = 0; i < tensor1D.len(); ++i) {
    EXPECT_DOUBLE_EQ(vec1D[i], tensor1D(i));
  }
}

///////////////////////////////////////////////////////////////////////
// Tensor Test 4
///////////////////////////////////////////////////////////////////////
TEST(Tensor, test4) {
  auto device = Device_T(0, 0);

  size_t N = 450;

  const auto vec1D = pysycl::generate_random_vector_1d(N, -100.0, 100.0);
  auto tensor1D = Tensor_T(device, vec1D);

  ASSERT_EQ(N, tensor1D.len());
  ASSERT_EQ(1, tensor1D.num_dims());

  for (int i = 0; i < tensor1D.len(); ++i) {
    EXPECT_DOUBLE_EQ(vec1D[i], tensor1D(i));
  }
}

///////////////////////////////////////////////////////////////////////
// Tensor Test 5
///////////////////////////////////////////////////////////////////////
TEST(Tensor, test5) {
  auto device = Device_T(0, 0);

  const int M = 4;
  const int N = 6;

  std::vector<std::vector<Scalar_T>> vec2D = {
      {1.853, -2.22, 0.213, 512.66, 1.6, -78.81},
      {-1.53, -9.12, 1.223, 10.12, 85.3, 3.128},
      {95.85, 58.12, 9.313, -1.66, 1.6, -99.87},
      {1.853, 69.22, 4.223, -15.77, -5.55, -55.13}};

  auto tensor2D = Tensor_T(device, vec2D);

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      EXPECT_DOUBLE_EQ(vec2D[i][j], tensor2D(i, j));
    }
  }
}

///////////////////////////////////////////////////////////////////////
// Tensor Test 6
///////////////////////////////////////////////////////////////////////
TEST(Tensor, test6) {
  auto device = Device_T(0, 0);

  const int M = 450;
  const int N = 675;

  const auto vec2D = pysycl::generate_random_vector_2d(M, N, -100.0, 100.0);

  auto tensor2D = Tensor_T(device, vec2D);

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      EXPECT_DOUBLE_EQ(vec2D[i][j], tensor2D(i, j));
    }
  }
}
#ifndef VECTOR_OPERATIONS_H
#define VECTOR_OPERATIONS_H

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
/// \brief Vector operations in PySYCL.
///////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////
// onemath
///////////////////////////////////////////////////////////////////////
#include <oneapi/mkl.hpp>

///////////////////////////////////////////////////////////////////////
// pybind
///////////////////////////////////////////////////////////////////////
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

///////////////////////////////////////////////////////////////////////
// sycl
///////////////////////////////////////////////////////////////////////
#include <sycl/sycl.hpp>

namespace py = pybind11;

namespace pysycl {
///////////////////////////////////////////////////////////////////////
/// \brief Function that computes a vector addition
template<typename Tensor_T>
auto vector_addition(Tensor_T& A, Tensor_T& B) {
  if(A.len() != B.len() || A.num_dims() != B.num_dims()) {
    throw std::runtime_error("ERROR in Vector Addition: Tensors of incompatible dimensions!");
  }

  if(A.device_reference() != B.device_reference()) {
    throw std::runtime_error("ERROR in Vector Addition: Tensors have incompatible device queues!");
  }

  auto& device = A.device_reference();
  const auto N = A.len();

  Tensor_T C = Tensor_T(device, py::make_tuple(N));

  device.get_queue().submit([&](sycl::handler& h) {
    auto* A_data = A.data_ptr();
    auto* B_data = B.data_ptr();
    auto* C_data = C.data_ptr();

    h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
      C_data[idx] = A_data[idx] + B_data[idx];
    });
  }).wait();

  return C;
}

} // namespace pysycl

#endif // VECTOR_OPERATIONS_H
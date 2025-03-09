#ifndef MATRIX_OPERATIONS_H
#define MATRIX_OPERATIONS_H

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
/// \brief Matrix operations in PySYCL.
///////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////
// onemath
///////////////////////////////////////////////////////////////////////
#include <oneapi/mkl.hpp>

///////////////////////////////////////////////////////////////////////
// sycl
///////////////////////////////////////////////////////////////////////
#include <sycl/sycl.hpp>

namespace pysycl {
///////////////////////////////////////////////////////////////////////
/// \brief Function that computes a matrix multiplication
template<typename Tensor_T>
auto matrix_multiplication(Tensor_T& A, Tensor_T& B) {
  using Scalar_T = Tensor_T::Scalar_T;

  if(A.num_dims() != B.num_dims() || A.num_dims() != 2) {
    throw std::runtime_error("ERROR in Matrix Multiplication: Tensors of incompatible dimensions!");
  }

  if(A.device_reference() != B.device_reference()) {
    throw std::runtime_error("ERROR in Matrix Multiplication: Tensors have incompatible device queues!");
  }

  auto& device = A.device_reference();

  const auto M = A.dims_list()[0];
  const auto K = A.dims_list()[1];
  const auto N = B.dims_list()[1];

  if(B.dims_list()[0] != K) {
    throw std::runtime_error("ERROR in Matrix Multiplication: Tensors of incompatible dimensions!");
  }

  Tensor_T C = Tensor_T(device, py::make_tuple(M, N));

  #ifdef PYSYCL_USE_ONEMKL
  oneapi::mkl::blas::column_major::gemm(
    device.get_queue(),
    oneapi::mkl::transpose::nontrans,
    oneapi::mkl::transpose::nontrans,
    M,
    N,
    K,
    1.0,
    A.data_ptr(),
    M,
    B.data_ptr(),
    K,
    1.0,
    C.data_ptr(),
    M).wait();
  #else
  device.get_queue().submit([&](sycl::handler &h){
    auto* A_data = A.data_ptr();
    auto* B_data = B.data_ptr();
    auto* C_data = C.data_ptr();

    h.parallel_for(sycl::range<2>(M, N), [=](sycl::id<2> idx){
      const auto i = idx[0];
      const auto j = idx[1];

      Scalar_T c_ij = 0.0;

      for(int k = 0; k < K; ++k){
        c_ij += A_data[k*M + i]*B_data[j*K + k];
      }
      C_data[j*M + i] = c_ij;
    });
  }).wait();
  #endif

  return C;
}

} // namespace pysycl

#endif // MATRIX_OPERATIONS_H
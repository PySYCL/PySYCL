#ifndef MATRIX_MULTIPLICATION_H
#define MATRIX_MULTIPLICATION_H

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
/// \brief Matrix Multiplication in PySYCL.
///////////////////////////////////////////////////////////////////////

/// Linear Algebra functionalities in PySYCL

///////////////////////////////////////////////////////////////////////
// sycl
///////////////////////////////////////////////////////////////////////
#include <CL/sycl.hpp>

///////////////////////////////////////////////////////////////////////
// local
///////////////////////////////////////////////////////////////////////
#include "../Device/Device_Instance.h"

///////////////////////////////////////////////////////////////////////
// \addtogroup Linear_Algebra
/// @{
namespace pysycl {

///////////////////////////////////////////////////////////////////////
/// \brief Tiled Matrix multiplication function.
/// \param[in] A The first Matrix that is being multiplied.
/// \param[in] B The second Matrix that is being multiplied.
/// \param[in] C The Matrix containing the result of the multiplication.
/// \param[in] wg_size The tile size.
template <typename Matrix_type>
void matmul(Matrix_type &A, Matrix_type &B, Matrix_type &C,
            const size_t &wg_size) {
  using Scalar_T = typename Matrix_type::Scalar_T;

  if (A.num_cols() != B.num_rows()) {
    throw std::runtime_error("ERROR: Incompatible Array2D dimensions.");
  }

  if (C.num_rows() != A.num_rows() || C.num_cols() != B.num_cols()) {
    throw std::runtime_error("ERROR: Incompatible Array2D dimensions.");
  }

  if(C.dev() != A.dev() || C.dev() != B.dev()) {
    throw std::runtime_error("ERROR: Incompatible PySYCL device.");
  }

  const auto M = A.num_rows();
  const auto N = A.num_cols();
  const auto P = B.num_cols();

  A.dev().get_queue().submit([&](sycl::handler &h) {
    const size_t global_size_M = ((M + wg_size - 1)/wg_size)*wg_size;
    const size_t global_size_P = ((P + wg_size - 1)/wg_size)*wg_size;

    sycl::range<2> global{global_size_M, global_size_P};
    sycl::range<2> local{wg_size, wg_size};

    Scalar_T *A_ptr = A.get_device_data_ptr();
    Scalar_T *B_ptr = B.get_device_data_ptr();
    Scalar_T *C_ptr = C.get_device_data_ptr();

    h.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> it) {
      int i = it.get_global_id(0);
      int j = it.get_global_id(1);

      Scalar_T c_ij = 0.0;

      if(i >= M || j >= P) return;

      for(int p = 0; p < N; ++p){
        c_ij += A_ptr[i*N + p]*B_ptr[p*P + j];
      }

      C_ptr[i*P + j] = c_ij;
    });
   }).wait();
}

/// @} // end "Linear_Algebra" doxygen group

} // namespace pysycl

#endif // MATRIX_MULTIPLICATION_H
#ifndef TRACE_H
#define TRACE_H

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
/// \brief Trace in PySYCL.
///////////////////////////////////////////////////////////////////////

/// General functionalities in PySYCL

///////////////////////////////////////////////////////////////////////
// sycl
///////////////////////////////////////////////////////////////////////
#include <CL/sycl.hpp>

///////////////////////////////////////////////////////////////////////
// local
///////////////////////////////////////////////////////////////////////
#include "../Device/Device_Instance.h"

///////////////////////////////////////////////////////////////////////
// \addtogroup Linear_ Algebra
/// @{
namespace pysycl {

///////////////////////////////////////////////////////////////////////
/// \brief Returns the trace of a matrix.
/// \param[in] A The input matrix.
/// \return The trace of the matrix.
template <typename Matrix_type>
auto trace(Matrix_type &A) {
  using Scalar_T = typename Matrix_type::Scalar_T;

  Scalar_T trace = 0.0;
  sycl::buffer<Scalar_T> trace_buf{&trace, 1};

  const size_t wg_size = A.dev().get_max_workgroup_size();

  const auto M = A.num_rows();
  const auto N = A.num_cols();

  if(M != N) {
    throw std::runtime_error("ERROR IN TRACE: Rows and columns must be of equal size.");
  }

  A.dev().get_queue().submit([&](sycl::handler &h) {
    const auto reduction_func = sycl::reduction(trace_buf, h, sycl::plus<Scalar_T>());
    const size_t global_size = ((N + wg_size - 1)/wg_size)*wg_size;

    sycl::range<1> global{global_size};
    sycl::range<1> local{wg_size};

    auto data = A.get_device_data_ptr();

    h.parallel_for(sycl::nd_range<1>(global, local), reduction_func,
                   [=](sycl::nd_item<1> it, auto& diag_el){
                    const auto idx = it.get_global_id();
                    if(idx >= N) return;

                    diag_el.combine(data[idx*N + idx]);
                   });
  }).wait();

  sycl::host_accessor trace_host{trace_buf, sycl::read_only};
  trace = trace_host[0];

  return trace;
}

/// @} // end "General" doxygen group

} // namespace pysycl

#endif // TRACE_H
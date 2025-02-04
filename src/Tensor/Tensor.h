#ifndef TENSOR_H
#define TENSOR_H

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
/// \brief Tensor in PySYCL.
///////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////
// local
///////////////////////////////////////////////////////////////////////
#include "../Device/Device_Instance.h"

///////////////////////////////////////////////////////////////////////
// sycl
///////////////////////////////////////////////////////////////////////
#include <sycl/sycl.hpp>

///////////////////////////////////////////////////////////////////////
// stl
///////////////////////////////////////////////////////////////////////
#include <vector>

///////////////////////////////////////////////////////////////////////
/// \addtogroup Device
/// @{

namespace pysycl {

///////////////////////////////////////////////////////////////////////
/// \brief Class defining a pysycl tensor.
template<typename Scalar_T>
class Tensor {
  public:
  /////////////////////////////////////////////////////////////////////
  /// \brief Copy constructor, use compiler generated version.
  Tensor(const Tensor&) = default;

  /////////////////////////////////////////////////////////////////////
  /// \brief Move constructor, use compiler generated version.
  Tensor(Tensor&&) = default;

  /////////////////////////////////////////////////////////////////////
  /// \brief Copy assignment, use compiler generated version.
  /// \return reference to the assigned object.
  Tensor& operator=(const Tensor&) = default;

  /////////////////////////////////////////////////////////////////////
  /// \brief Move assignment, use compiler generated version.
  /// \return reference to the assigned object.
  Tensor& operator=(Tensor&&) = default;

  /////////////////////////////////////////////////////////////////////
  /// \brief Constructor that creates a pysycl tensor.
  /// \param[in] device_in device that the memory resides on.
  /// \param[in] dimensions_in dimensions for the tensor.
  Tensor(const Device_Instance& device_in,
         const std::vector<Scalar_T>& dimension_sizes_in)
    : device(device_in)
    , dimension_sizes(dimension_sizes_in)
    , dimensions(dimension_sizes.size()) {
      for(const auto& dimension_size : dimension_sizes) {
        total_length *= dimension_size;
      }

      Scalar_T* data = sycl::malloc_shared<Scalar_T>(total_length, Q);
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Get the number of elements in the Vector.
  /// \return Number of elements in the Vector.
  int len() const { return total_length; }

  private:
  ///////////////////////////////////////////////////////////////////////
  /// \brief The device that will load the usm memory.
  Device_Instance device;

  ///////////////////////////////////////////////////////////////////////
  /// \brief The length of the tensor in each dimension.
  std::vector<Scalar_T> dimension_sizes;

  ///////////////////////////////////////////////////////////////////////
  /// \brief The number of dimensions.
  std::size_t dimensions;

  ///////////////////////////////////////////////////////////////////////
  /// \brief The total length of the memory
  std::size_t total_length = 1.0;

  ///////////////////////////////////////////////////////////////////////
  /// \brief The pointer to usm memory
  Scalar_T* data;
};

} // namespace pysycl

#endif // #ifndef TENSOR_H
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
#include "../Device/Device.h"

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
  Tensor(const Device& device_in,
         const std::vector<std::size_t>& dims_in)
    : device(device_in)
    , dims(dims_in) {
      for(const auto& dim : dims) {
        if(dim == 0) throw std::runtime_error("ERROR in Tensor: Cannot have dimension of zero length");
        length *= dim;
      }

      Scalar_T* data = sycl::malloc_shared<Scalar_T>(length, device.get_queue());
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Get the number of elements in the tensor.
  /// \return Number of elements in the tensor.
  std::size_t len() const { return length; }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Get the number of dimensions in the tensor.
  /// \return Number of dimensions in the tensor.
  std::size_t num_dims() const { return dims.size(); }

  private:
  ///////////////////////////////////////////////////////////////////////
  /// \brief The device that will load the usm memory.
  Device device;

  ///////////////////////////////////////////////////////////////////////
  /// \brief The length of the tensor in each dimension.
  std::vector<std::size_t> dims;

  ///////////////////////////////////////////////////////////////////////
  /// \brief The total length of the memory
  std::size_t length = 1.0;

  ///////////////////////////////////////////////////////////////////////
  /// \brief The pointer to usm memory
  Scalar_T* data;
};

} // namespace pysycl

#endif // #ifndef TENSOR_H
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
// pybind
///////////////////////////////////////////////////////////////////////
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

///////////////////////////////////////////////////////////////////////
// sycl
///////////////////////////////////////////////////////////////////////
#include <sycl/sycl.hpp>

///////////////////////////////////////////////////////////////////////
// stl
///////////////////////////////////////////////////////////////////////
#include <vector>

namespace py = pybind11;

///////////////////////////////////////////////////////////////////////
/// \addtogroup Tensor
/// @{

namespace pysycl {

///////////////////////////////////////////////////////////////////////
/// \brief Class defining a pysycl tensor.
template<typename Scalar_type>
class Tensor {
  public:
  /////////////////////////////////////////////////////////////////////
  /// \brief The scalar type that is used.
  using Scalar_T = Scalar_type;

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
  /// \brief Constructor that creates a pysycl tensor based on
  ///        dimensional parameters.
  /// \tparam Dimensions variadic parameter pack of input dimensions
  /// \param[in] device_in device that the memory resides on.
  /// \param[in] dims_in dimensions for the tensor.
  Tensor(const Device& device_in,
         const py::tuple& dims_in)
    : device(device_in) {

      for(int i = 0; i < py::len(dims_in); ++i) {
        dims.push_back(dims_in[i].cast<size_t>());

        if(dims[i] == 0) {
          throw std::runtime_error("ERROR in Tensor: Cannot have zero dimension!");
        }

        length *= dims[i];
      }

      data = sycl::malloc_shared<Scalar_T>(length, device.get_queue());
  }

  /////////////////////////////////////////////////////////////////////
  /// \brief Constructor that creates an ND pysycl tensor based on
  ///        dimensional parameters.
  /// \tparam Dimensions variadic parameter pack of input dimensions
  /// \param[in] device_in device that the memory resides on.
  /// \param[in] dimensions_in dimensions for the tensor.
  /// \param[in] data_in data input for the tensor.
  Tensor(const Device& device_in,
         const std::vector<Scalar_T>& data_in,
         const py::tuple& dims_in)
    : device(device_in) {

      for(int i = 0; i < py::len(dims_in); ++i) {
        dims.push_back(dims_in[i].cast<size_t>());

        if(dims[i] == 0) {
          throw std::runtime_error("ERROR in Tensor: Cannot have zero dimension!");
        }

        length *= dims[i];
      }

      data = sycl::malloc_shared<Scalar_T>(length, device.get_queue());

      for(int i = 0; i < data_in.size(); ++i) {
        data[i] = data_in[i];
      }
  }

  /////////////////////////////////////////////////////////////////////
  /// \brief Constructor that creates a 1D pysycl tensor
  /// \param[in] device_in device that the memory resides on.
  /// \param[in] data_in data input for the tensor.
  Tensor(const Device& device_in,
         const std::vector<Scalar_T>& data_in)
    : device(device_in),
      dims({data_in.size()}) {

      if(data_in.size() == 0) {
        throw std::runtime_error("ERROR in Tensor: Input data is empty!");
      }

      length = data_in.size();

      data = sycl::malloc_shared<Scalar_T>(length, device.get_queue());

      for(int i = 0; i < data_in.size(); ++i) {
        data[i] = data_in[i];
      }
  }

  /////////////////////////////////////////////////////////////////////
  /// \brief Constructor that creates a 2D pysycl tensor
  /// \param[in] device_in device that the memory resides on.
  /// \param[in] data_in data input for the tensor.
  Tensor(const Device& device_in,
         const std::vector<std::vector<Scalar_T>>& data_in)
    : device(device_in),
      dims({data_in.size(), data_in[0].size()}) {

      if(data_in.size() == 0) {
        throw std::runtime_error("ERROR in Tensor: Input data is empty!");
      }

      for(int i = 0; i < data_in.size(); ++i) {
        if(data_in[i].size() != data_in[0].size()) {
          throw std::runtime_error("ERROR in Tensor: Input vector has invalid dimensions!");
        }
      }

      length = data_in.size() * data_in[0].size();

      data = sycl::malloc_shared<Scalar_T>(length, device.get_queue());

      for(size_t i = 0; i < data_in.size(); ++i) {
        for(size_t j = 0; j < data_in[0].size(); ++j) {
          data[global_index({i, j})] = data_in[i][j];
        }
      }
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Get the number of elements in the tensor.
  /// \return Number of elements in the tensor.
  size_t len() const {
    return length;
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Get the number of dimensions in the tensor.
  /// \return Number of dimensions in the tensor.
  size_t num_dims() const {
    return dims.size();
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Get the number of dimensions in the tensor.
  /// \return Number of dimensions in the tensor.
  void fill(Scalar_T& val) const {
    for(int i = 0; i < length; ++i) {
      data[i] = val;
    }
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Overloaded operator for direct element access.
  /// \return Number of dimensions in the tensor.
  size_t global_index(std::vector<size_t> index_list) {
    if(index_list.size() != dims.size() || index_list.size() <= 0) {
      throw std::runtime_error("ERROR in Tensor: Invalid number of indices!");
    }

    size_t idx = 0;

    size_t multiplier = 1;

    for(int i = index_list.size() - 1; i >= 0; --i) {
      idx += index_list[i] * multiplier;
      multiplier *= dims[i];
    }

    return idx;
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Function that returns the device.
  /// \return Reference to the device.
  auto &device_reference() {
    return device;
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Function that returns the data.
  /// \return Reference to the data.
  Scalar_T* data_ptr() {
    return data;
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Function that returns the list of dimensions.
  /// \return The list of dimensions.
  auto dims_list() {
    return dims;
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Overloaded operator for direct element access.
  /// \return Number of dimensions in the tensor.
  template<typename... Indices>
  Scalar_T &operator()(Indices... indices) {
    std::vector<size_t> index_list = {static_cast<size_t>(indices)...};
    return data[global_index(index_list)];
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Overloaded operator for direct element access.
  /// \return Number of dimensions in the tensor.
  Scalar_T &operator()(py::tuple& indices) {
    std::vector<size_t> index_list;

    for(int i = 0; i < py::len(indices); ++i) {
      index_list.push_back(indices[i].cast<size_t>());
    }

    return data[global_index(index_list)];
  }

  private:
  ///////////////////////////////////////////////////////////////////////
  /// \brief The device that will load the usm memory.
  Device device;

  ///////////////////////////////////////////////////////////////////////
  /// \brief The length of the tensor in each dimension.
  std::vector<size_t> dims;

  ///////////////////////////////////////////////////////////////////////
  /// \brief The total length of the memory
  size_t length = 1;

  ///////////////////////////////////////////////////////////////////////
  /// \brief The pointer to usm memory
  Scalar_T* data;
};

} // namespace pysycl

#endif // #ifndef TENSOR_H
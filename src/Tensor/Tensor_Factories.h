#ifndef TENSOR_FACTORIES_H
#define TENSOR_FACTORIES_H

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
/// \brief Generate Tensors in PySYCL
///////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////
// local
///////////////////////////////////////////////////////////////////////
#include "../Device/Device.h"
#include "../Data_Types/Data_Types.h"
#include "Tensor.h"

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
#include <algorithm>
#include <vector>

namespace py = pybind11;

///////////////////////////////////////////////////////////////////////
/// \addtogroup Tensor
/// @{

namespace pysycl {

using Tensor_double_T = pysycl::Tensor<double>;
using Tensor_int_T    = pysycl::Tensor<int>;
using Tensor_float_T  = pysycl::Tensor<float>;

using Tensor_Variant_T = std::variant<Tensor_double_T, Tensor_int_T, Tensor_float_T>;

/////////////////////////////////////////////////////////////////////
/// \brief Factory to construct a dimensionless pysycl tensor.
/// \tparam Dimensions variadic parameter pack of input dimensions
/// \param[in] device device that the memory resides on.
/// \param[in] dtype  the data type for the tensor.
Tensor_Variant_T tensor_factories(const Device& device,
                                  const Data_Types& dtype) {
  if(dtype == Data_Types::FLOAT64) {
    return Tensor<double>(device);
  } else if(dtype == Data_Types::FLOAT32) {
    return Tensor<float>(device);
  } else if(dtype == Data_Types::INT16) {
    return Tensor<int>(device);
  } else {
    throw std::runtime_error("ERROR IN TENSOR FACTORIES: Unsupported data type!");
  }
}

/////////////////////////////////////////////////////////////////////
/// \brief Factory to construct a dimensionless pysycl tensor.
/// \param[in] device device that the memory resides on.
/// \param[in] dims device that the memory resides on.
/// \param[in] dtype     the data type for the tensor.
Tensor_Variant_T tensor_factories(const Device& device,
                                  const py::tuple& dims,
                                  const Data_Types& dtype) {
  if(dtype == Data_Types::FLOAT64) {
    return Tensor<double>(device, dims);
  } else if(dtype == Data_Types::FLOAT32) {
    return Tensor<float>(device, dims);
  } else if(dtype == Data_Types::INT16) {
    return Tensor<int>(device, dims);
  } else {
    throw std::runtime_error("ERROR IN TENSOR FACTORIES: Unsupported data type!");
  }
}

/////////////////////////////////////////////////////////////////////
/// \brief Factory to construct a 1D pysycl tensor with data
///        from an input vector
/// \param[in] device device that the memory resides on.
/// \param[in] data   input data.
/// \param[in] dtype  the data type for the tensor.
template<typename Scalar_T>
Tensor_Variant_T tensor_factories(const Device& device,
                                  const std::vector<Scalar_T>& data,
                                  const Data_Types& dtype) {
  if(dtype == Data_Types::FLOAT64) {
    std::vector<double> type_data(data.size(), 0);
    std::copy(data.begin(), data.end(), type_data.begin());

    return Tensor<double>(device, type_data);
  } else if(dtype == Data_Types::FLOAT32) {
    std::vector<float> type_data(data.size(), 0);
    std::copy(data.begin(), data.end(), type_data.begin());

    return Tensor<float>(device, type_data);
  } else if(dtype == Data_Types::INT16) {
    std::vector<int> type_data(data.size(), 0);
    std::copy(data.begin(), data.end(), type_data.begin());

    return Tensor<int>(device, type_data);
  } else {
    throw std::runtime_error("ERROR IN TENSOR FACTORIES: Unsupported data type!");
  }
}

/////////////////////////////////////////////////////////////////////
/// \brief Factory to construct a 2D pysycl tensor with data
///        from an input vector
/// \param[in] device device that the memory resides on.
/// \param[in] data   input data.
/// \param[in] dtype  the data type for the tensor.
template<typename Scalar_T>
Tensor_Variant_T tensor_factories(const Device& device,
                                  const std::vector<std::vector<Scalar_T>>& data,
                                  const Data_Types& dtype) {

  if(dtype == Data_Types::FLOAT64) {
    std::vector<std::vector<double>> type_data(data.size());

    std::transform(data.begin(), data.end(), type_data.begin(), [](const auto& data_in) {
      return std::vector<double>(data_in.begin(), data_in.end());
    });

    return Tensor<double>(device, type_data);
  } else if(dtype == Data_Types::FLOAT32) {
    std::vector<std::vector<float>> type_data(data.size());

    std::transform(data.begin(), data.end(), type_data.begin(), [](const auto& data_in) {
      return std::vector<float>(data_in.begin(), data_in.end());
    });

    return Tensor<float>(device, type_data);
  } else if(dtype == Data_Types::INT16) {
    std::vector<std::vector<int>> type_data(data.size());

    std::transform(data.begin(), data.end(), type_data.begin(), [](const auto& data_in) {
      return std::vector<int>(data_in.begin(), data_in.end());
    });

    return Tensor<int>(device, type_data);
  } else {
    throw std::runtime_error("ERROR IN TENSOR FACTORIES: Unsupported data type!");
  }
}

} // namespace pysycl

#endif // #ifndef TENSOR_FACTORIES_H
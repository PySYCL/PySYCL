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
// sycl
///////////////////////////////////////////////////////////////////////
#include <sycl/sycl.hpp>

///////////////////////////////////////////////////////////////////////
// stl
///////////////////////////////////////////////////////////////////////
#include <vector>

///////////////////////////////////////////////////////////////////////
/// \addtogroup Tensor
/// @{

namespace pysycl {

using Tensor_double_T = pysycl::Tensor<double>;
using Tensor_int_T    = pysycl::Tensor<int>;
using Tensor_float_T  = pysycl::Tensor<float>;

using Tensor_Variant_T = std::variant<Tensor_double_T, Tensor_int_T, Tensor_float_T>;

Tensor_Variant_T tensor_factories(Device_Instance &device, Data_Types &dtype) {
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

} // namespace pysycl

#endif // #ifndef TENSOR_FACTORIES_H
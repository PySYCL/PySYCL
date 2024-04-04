#ifndef VECTOR_FACTORIES_H
#define VECTOR_FACTORIES_H

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
/// \brief PySYCL Vector Factories.
///////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////
/// local
///////////////////////////////////////////////////////////////////////
#include "../Data_Types/Data_Types.h"
#include "../Device/Device_Instance.h"
#include "../Device/Device_Manager.h"
#include "Vector_Type.h"

///////////////////////////////////////////////////////////////////////
// pybind11
///////////////////////////////////////////////////////////////////////
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

///////////////////////////////////////////////////////////////////////
/// stl
///////////////////////////////////////////////////////////////////////
#include <tuple>
#include <variant>

namespace py = pybind11;

using Device_T = pysycl::Device_Instance;
using Data_T = pysycl::Data_Types;

///////////////////////////////////////////////////////////////////////
/// \addtogroup Vector
/// @{
namespace pysycl {

using Vector_Variants = std::variant<Vector<double>,
                                     Vector<float>,
                                     Vector<int>>;

///////////////////////////////////////////////////////////////////////
/// \brief Function factory for Vector Types.
/// \param[in] dims The dimension of the vector.
/// \param[in] device_in The target sycl device.
/// \param[in] dtype The data type of the vector.
Vector_Variants
vector_factories(int dims, Device_Instance &device, Data_Types& dtype) {
  if(dtype == Data_Types::DOUBLE) {
    return Vector<double>(dims, device);
  } else if(dtype == Data_Types::FLOAT) {
    return Vector<float>(dims, device);
  } else if (dtype == Data_Types::INT) {
    return Vector<int>(dims, device);
  } else {
    throw std::runtime_error("ERROR IN VECTOR: Unsupported data type.");
  }
}

///////////////////////////////////////////////////////////////////////
/// \brief Function factory for Vector Types with input numpy array.
/// \param[in] np_array The input numpy array.
/// \param[in] device The target sycl device.
/// \param[in] dtype The data type of the vector.
template<typename Scalar_T>
Vector_Variants
vector_factories(py::array_t<Scalar_T> np_array,
                 Device_Instance &device,
                 Data_Types& dtype) {
  if(dtype == Data_Types::DOUBLE) {
    return Vector<double>(np_array, device);
  } else if(dtype == Data_Types::FLOAT) {
    return Vector<float>(np_array, device);
  } else if (dtype == Data_Types::INT) {
    return Vector<int>(np_array, device);
  } else {
    throw std::runtime_error("ERROR IN VECTOR: Unsupported data type.");
  }
}

/// @} // end "Vector" doxygen group

} // namespace pysycl

#endif // VECTOR_FACTORIES_H

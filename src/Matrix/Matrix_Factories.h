#ifndef MATRIX_FACTORIES_H
#define MATRIX_FACTORIES_H

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
/// \brief PySYCL Matrix Factories.
///////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////
/// stl
///////////////////////////////////////////////////////////////////////
#include <tuple>
#include <variant>

///////////////////////////////////////////////////////////////////////
/// local
///////////////////////////////////////////////////////////////////////
#include "../Data_Types/Data_Types.h"
#include "../Device/Device_Instance.h"
#include "../Device/Device_Manager.h"
#include "Matrix_Type.h"

namespace py = pybind11;

using Device_T = pysycl::Device_Instance;
using Data_T = pysycl::Data_Types;

///////////////////////////////////////////////////////////////////////
/// \addtogroup Matrix
/// @{
namespace pysycl {

using Matrix_Variants = std::variant<Matrix<double>,
                                     Matrix<float>,
                                     Matrix<int>>;

///////////////////////////////////////////////////////////////////////
/// \brief Function specialization for Matrix.
/// \param[in] dims The dimension of the matrix.
/// \param[in] device_in The target sycl device.
/// \param[in] dtype The data type of the matrix.
Matrix_Variants
matrix_factories(std::tuple<int, int> dims, Device_Instance &device, Data_Types& dtype) {
  if(dtype == Data_Types::DOUBLE) {
    return Matrix<double>(std::get<0>(dims), std::get<1>(dims), device);
  } else if(dtype == Data_Types::FLOAT) {
    return Matrix<float>(std::get<0>(dims), std::get<1>(dims), device);
  } else if (dtype == Data_Types::INT) {
    return Matrix<int>(std::get<0>(dims), std::get<1>(dims), device);
  } else {
    throw std::runtime_error("ERROR IN MATRIX: Unsupported datatype.");
  }
}

///////////////////////////////////////////////////////////////////////
/// \brief Function factory for Matrix Types with input numpy array.
/// \param[in] np_array The input numpy array.
/// \param[in] device The target sycl device.
/// \param[in] dtype The data type of the vector.
template<typename Scalar_T>
Matrix_Variants
matrix_factories(py::array_t<Scalar_T> np_array,
                 Device_Instance &device,
                 Data_Types& dtype) {
  if(dtype == Data_Types::DOUBLE) {
    return Matrix<double>(np_array, device);
  } else if(dtype == Data_Types::FLOAT) {
    return Matrix<float>(np_array, device);
  } else if (dtype == Data_Types::INT) {
    return Matrix<int>(np_array, device);
  } else {
    throw std::runtime_error("ERROR IN MATRIX: Unsupported data type.");
  }
}

/// @} // end "Matrix" doxygen group

} // namespace pysycl

#endif // MATRIX_FACTORIES_H
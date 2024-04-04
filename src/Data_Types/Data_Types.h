#ifndef DATA_TYPES_H
#define DATA_TYPES_H

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

/// Data Types in PySYCL

///////////////////////////////////////////////////////////////////////
/// \file
/// \brief Data types in PySYCL.
///////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////
/// \addtogroup Data Types
/// @{
namespace pysycl {

///////////////////////////////////////////////////////////////////////
/// \brief Data types
enum class Data_Types { DOUBLE, FLOAT, INT };

/// @} // end "Data_Types" doxygen group

} // namespace pysycl

#endif // DATA_TYPES_H
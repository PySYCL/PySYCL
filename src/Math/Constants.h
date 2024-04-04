#ifndef CONSTANTS_H
#define CONSTANTS_H

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
/// \brief Math related constants in PySYCL.
///////////////////////////////////////////////////////////////////////

/// Math related matter in PySYCL

///////////////////////////////////////////////////////////////////////
// stl
///////////////////////////////////////////////////////////////////////
#include <cmath>

///////////////////////////////////////////////////////////////////////
/// \addtogroup Math
/// @{
namespace pysycl {
///////////////////////////////////////////////////////////////////////
/// \brief PI with 61 decimal places
constexpr static double PI = 3.1415926535897932384626433832795028841971693993751058209749445;
/// @} // end "Math" doxygen group

} // namespace pysycl

#endif // CONSTANTS_H
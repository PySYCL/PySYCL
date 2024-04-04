#ifndef BIT_MANIPULATION_FUNCTIONS_H
#define BIT_MANIPULATION_FUNCTIONS_H

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
/// \brief Bit Manipulation Functions.
///////////////////////////////////////////////////////////////////////

/// Functions for Bit Manipulation in PySYCL

///////////////////////////////////////////////////////////////////////
// stl
///////////////////////////////////////////////////////////////////////
#include <cmath>

///////////////////////////////////////////////////////////////////////
// \addtogroup Bit_Manipulation
/// @{
namespace pysycl {

///////////////////////////////////////////////////////////////////////
/// \brief Reverses the bits of the input number.
/// \return The number with reversed bits.
template <typename Int_T> Int_T bit_reverse(Int_T N) {
  Int_T N_r = 0;

  for (Int_T i = 0; i < sizeof(N) * 8; ++i) {
    N_r <<= 1;
    N_r |= (N & 1);
    N >>= 1;
  }

  return N_r;
}

///////////////////////////////////////////////////////////////////////
/// \brief Finds the next power of 2 larger than N.
/// \return Next power of 2.
template <typename Int_T> Int_T next_power_of_two(Int_T N) {
  if (N <= 0)
    return 1;
  --N;

  N |= N >> 1;
  N |= N >> 2;
  N |= N >> 4;
  N |= N >> 8;
  N |= N >> 16;

  ++N;

  return N;
}

/// @} // end "Bit_Manipulation" doxygen group

} // namespace pysycl

#endif // BIT_MANIPULATION_FUNCTIONS_H
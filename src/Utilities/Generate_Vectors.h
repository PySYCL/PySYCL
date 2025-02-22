#ifndef GENERATE_VECTORS_H
#define GENERATE_VECTORS_H

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
/// \brief Generate vectors in PySYCL.
///////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////
// stl
///////////////////////////////////////////////////////////////////////
#include <bits/stdc++.h>

namespace pysycl {
///////////////////////////////////////////////////////////////////////
/// \brief Function that generates a 1D vector with random values
template<typename Scalar_T>
auto generate_random_vector_1d(int N, Scalar_T min, Scalar_T max) {
  std::vector<Scalar_T> vec(N, 0.0);

  std::random_device random_device;
  std::mt19937 generator(random_device());

  if constexpr (std::is_integral<Scalar_T>::value) {
    std::uniform_int_distribution<Scalar_T> distribution(min, max);

    for(int i = 0; i < N; ++i) {
      vec[i] = distribution(generator);
    }

  } else {
    std::uniform_real_distribution<Scalar_T> distribution(min, max);

    for(int i = 0; i < N; ++i) {
      vec[i] = distribution(generator);
    }
  }

  return vec;
}

///////////////////////////////////////////////////////////////////////
/// \brief Function that generates a 1D vector with random values
template<typename Scalar_T>
auto generate_random_vector_2d(int M, int N, Scalar_T min, Scalar_T max) {
  std::vector<std::vector<Scalar_T>> vec(M, std::vector<Scalar_T>(N, 0));

  std::random_device random_device;
  std::mt19937 generator(random_device());

  if constexpr (std::is_integral<Scalar_T>::value) {
    std::uniform_int_distribution<Scalar_T> distribution(min, max);

    for(int i = 0; i < M; ++i) {
      for(int j = 0; j < N; ++j) {
        vec[i][j] = distribution(generator);
      }
    }
  } else {
    std::uniform_real_distribution<Scalar_T> distribution(min, max);

    for(int i = 0; i < M; ++i) {
      for(int j = 0; j < N; ++j) {
        vec[i][j] = distribution(generator);
      }
    }
  }

  return vec;
}

} // namespace pysycl

#endif // GENERATE_VECTORS_H
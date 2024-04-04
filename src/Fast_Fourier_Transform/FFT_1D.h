#ifndef FFT_1D_H
#define FFT_1D_H

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
/// \brief 1-Dimensional Fast Fourier Transform.
///////////////////////////////////////////////////////////////////////

/// FFT Functions in PySYCL

///////////////////////////////////////////////////////////////////////
// local
///////////////////////////////////////////////////////////////////////
#include "../Bit_Manipulation/Bit_Manipulation_Functions.h"
#include "../Device/Device_Instance.h"
#include "../Math/Constants.h"

///////////////////////////////////////////////////////////////////////
// stl
///////////////////////////////////////////////////////////////////////
#include <cmath>
#include <complex>
#include <vector>

///////////////////////////////////////////////////////////////////////
// sycl
///////////////////////////////////////////////////////////////////////
#include <CL/sycl.hpp>

///////////////////////////////////////////////////////////////////////
// \addtogroup Fast_Fourier_Transform
/// @{
namespace pysycl {

///////////////////////////////////////////////////////////////////////
/// \brief Function to check if the input value is a power of two.
/// \param[in] N The input value.
/// \return If the function is a power of two.
bool is_power_of_two(int N) {
  if (N <= 0)
    return false;
  return (N & (N - 1)) == 0;
}

///////////////////////////////////////////////////////////////////////
/// \brief 1-dimensional fast fourier transform (cooley-turkey) kernel.
/// This implementation adapts the CUDA FFT approach discussed in
/// the GitHub repository: [roguh/cuda-fft](https://github.com/roguh/cuda-fft).
/// \param[in] A The input vector for fft.
/// \param[in] result The result vector for fft.
/// \param[in] Q The sycl queue.
template <typename Input_T, typename Output_T, typename Int_T>
auto cooley_turkey_kernel(Input_T *data, Output_T *result_device, Int_T N, Int_T logn,
                   Int_T wg_size, sycl::queue Q) {
  Q.submit([&](sycl::handler &h) {
     const size_t global_size = ((N / 2 + wg_size - 1) / wg_size) * wg_size;

     sycl::range<1> global{global_size};
     sycl::range<1> local{wg_size};

     h.parallel_for(sycl::nd_range<1>(global, local), [=](sycl::nd_item<1> it) {
       const Int_T i = it.get_global_id(0);

       Int_T rev;

       rev = bit_reverse(2 * i);
       rev = rev >> (32 - logn);
       result_device[2 * i] = data[rev];

       rev = bit_reverse(2 * i + 1);
       rev = rev >> (32 - logn);
       result_device[2 * i + 1] = data[rev];

       it.barrier(sycl::access::fence_space::local_space);

       for (Int_T s = 1; s <= logn; ++s) {
         Int_T mh = 1 << (s - 1);
         Int_T k = (i / mh) * (1 << s);
         Int_T j = i % mh;
         Int_T kj = k + j;

         auto a = result_device[kj];
         auto b = result_device[kj + mh];

         float theta = -PI * j / mh;
         Output_T twiddle(std::cos(theta), std::sin(theta));
         auto b_twiddle = twiddle * b;

         result_device[kj] = a + b_twiddle;
         result_device[kj + mh] = a - b_twiddle;

         it.barrier(sycl::access::fence_space::local_space);
       }
     });
   }).wait();
}

///////////////////////////////////////////////////////////////////////
/// \brief 1-dimensional fast fourier transform function (cooley-turkey).
/// This implementation adapts the CUDA FFT approach discussed in
/// the GitHub repository: [roguh/cuda-fft](https://github.com/roguh/cuda-fft).
/// \param[in] A The input vector for fft.
/// \return The result of the fft
template <typename Vector_type> auto cooley_turkey(Vector_type &A) {
  using Scalar_T = typename Vector_type::Scalar_T;
  using Int_T = uint32_t;

  const Int_T N = A.get_size();
  const Int_T logn = std::log2(N);

  const Int_T wg_size = A.dev().get_max_workgroup_size();

  std::vector<std::complex<Scalar_T>> result_host(N);

  std::complex<Scalar_T> *result_device =
      sycl::malloc_device<std::complex<Scalar_T>>(N, A.dev().get_queue());

  Scalar_T *data = A.get_device_data_ptr();
  sycl::queue &Q = A.dev().get_queue();

  cooley_turkey_kernel(data, result_device, N, logn, wg_size, Q);

  A.dev()
      .get_queue()
      .memcpy(result_host.data(), result_device,
              N * sizeof(std::complex<Scalar_T>))
      .wait();
  return result_host;
}

///////////////////////////////////////////////////////////////////////
/// \brief 1-dimensional fast fourier transform function (chirp-z).
/// Reference for the Chirp Z-transform implementation, followed
/// [Chirp Z-transform](https://en.wikipedia.org/wiki/Chirp_Z-transform).
/// \param[in] A The input vector for fft.
/// \return The result of the fft
template <typename Vector_type> auto chirpz(Vector_type &A) {
  using Scalar_T = typename Vector_type::Scalar_T;
  using Int_T = uint32_t;

  const Int_T N = A.get_size();

  // Length for zero padding
  Int_T M = 1;
  while (M < 2 * N - 1)
    M *= 2;

  const Int_T wg_size = A.dev().get_max_workgroup_size();

  std::vector<std::complex<Scalar_T>> result_host(N);

  Scalar_T *data = A.get_device_data_ptr();
  sycl::queue &Q = A.dev().get_queue();

  std::complex<Scalar_T> *a_n =
      sycl::malloc_device<std::complex<Scalar_T>>(M, Q);
  std::complex<Scalar_T> *b_n =
      sycl::malloc_device<std::complex<Scalar_T>>(M, Q);
  std::complex<Scalar_T> *a_n_star =
      sycl::malloc_device<std::complex<Scalar_T>>(M, Q);
  std::complex<Scalar_T> *b_n_star =
      sycl::malloc_device<std::complex<Scalar_T>>(M, Q);

  std::complex<Scalar_T> *result_device =
      sycl::malloc_device<std::complex<Scalar_T>>(N, Q);

  // Sequences a and b
  Q.submit([&](sycl::handler &h) {
     const size_t global_size = ((M + wg_size - 1) / wg_size) * wg_size;

     sycl::range<1> global{global_size};
     sycl::range<1> local{wg_size};

     h.parallel_for(sycl::nd_range<1>(global, local), [=](sycl::nd_item<1> it) {
       const Int_T i = it.get_global_id(0);

       if (i < N) {
         const auto theta = PI * i * i / N;
         a_n[i] = data[i] * std::complex<Scalar_T>(cos(-theta), sin(-theta));
         b_n[i] = std::complex<Scalar_T>(cos(theta), sin(theta));
       } else if (i >= N && i < M - N + 1) {
         a_n[i] = std::complex<Scalar_T>(0.0, 0.0);
         b_n[i] = std::complex<Scalar_T>(0.0, 0.0);
       } else {
         const auto theta = PI * (M - i) * (M - i) / N;
         a_n[i] = std::complex<Scalar_T>(0.0, 0.0);
         b_n[i] = std::complex<Scalar_T>(cos(theta), sin(theta));
       }
     });
   }).wait();

  const Int_T logm = std::log2(M);
  cooley_turkey_kernel(a_n, a_n_star, M, logm, wg_size, Q);
  cooley_turkey_kernel(b_n, b_n_star, M, logm, wg_size, Q);

  Q.submit([&](sycl::handler &h) {
     const size_t global_size = ((M + wg_size - 1) / wg_size) * wg_size;

     sycl::range<1> global{global_size};
     sycl::range<1> local{wg_size};

     h.parallel_for(sycl::nd_range<1>(global, local), [=](sycl::nd_item<1> it) {
       const Int_T i = it.get_global_id(0);

       a_n[i] = a_n_star[i] * b_n_star[i];
       b_n[i] = b_n_star[i];

       a_n_star[i] = std::complex<Scalar_T>(0.0, 0.0);
       b_n_star[i] = std::complex<Scalar_T>(0.0, 0.0);
     });
   }).wait();

  // inverse fft a
  Q.submit([&](sycl::handler &h) {
     const size_t global_size = ((M + wg_size - 1) / wg_size) * wg_size;

     sycl::range<1> global{global_size};
     sycl::range<1> local{wg_size};

     h.parallel_for(sycl::nd_range<1>(global, local), [=](sycl::nd_item<1> it) {
       const Int_T i = it.get_global_id(0);

       a_n[i] = conj(a_n[i]);
     });
   }).wait();

  cooley_turkey_kernel(a_n, a_n_star, M, logm, wg_size, Q);

  Q.submit([&](sycl::handler &h) {
     const size_t global_size = ((M + wg_size - 1) / wg_size) * wg_size;

     sycl::range<1> global{global_size};
     sycl::range<1> local{wg_size};

     h.parallel_for(sycl::nd_range<1>(global, local), [=](sycl::nd_item<1> it) {
       const Int_T i = it.get_global_id(0);

       a_n[i] = conj(a_n_star[i]) / static_cast<Scalar_T>(M);
     });
   }).wait();

  Q.submit([&](sycl::handler &h) {
     const size_t global_size = ((N + wg_size - 1) / wg_size) * wg_size;

     sycl::range<1> global{global_size};
     sycl::range<1> local{wg_size};

     h.parallel_for(sycl::nd_range<1>(global, local), [=](sycl::nd_item<1> it) {
       const Int_T i = it.get_global_id(0);

       const auto theta = -PI * i * i / N;
       result_device[i] =
           a_n[i] * std::complex<Scalar_T>(cos(theta), sin(theta));
     });
   }).wait();

  Q.memcpy(result_host.data(), result_device,
           N * sizeof(std::complex<Scalar_T>))
      .wait();

  sycl::free(a_n, Q);
  sycl::free(b_n, Q);
  sycl::free(a_n_star, Q);
  sycl::free(b_n_star, Q);
  sycl::free(result_device, Q);

  return result_host;
}

///////////////////////////////////////////////////////////////////////
/// \brief 1-dimensional fast fourier transform function.
/// \param[in] A The input vector for fft.
/// \return The result of the fft
template <typename Vector_type> auto fft1d(Vector_type &A) {
  const auto N = A.get_size();

  if (is_power_of_two(N))
    return cooley_turkey(A);
  else
    return chirpz(A);
}

/// @} // end "Fast_Fourier_Transform" doxygen group

} // namespace pysycl

#endif // FFT_1D_H

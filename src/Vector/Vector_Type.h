#ifndef VECTOR_TYPE_H
#define VECTOR_TYPE_H

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
/// \brief Vector Type Object in PySYCL.
///////////////////////////////////////////////////////////////////////

/// 1-Dimensional Vectors in PySYCL

///////////////////////////////////////////////////////////////////////
// sycl
///////////////////////////////////////////////////////////////////////
#include <CL/sycl.hpp>

///////////////////////////////////////////////////////////////////////
// pybind11
///////////////////////////////////////////////////////////////////////
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

///////////////////////////////////////////////////////////////////////
// stl
///////////////////////////////////////////////////////////////////////
#include <cmath>
#include <vector>

///////////////////////////////////////////////////////////////////////
// local
///////////////////////////////////////////////////////////////////////
#include "../Device/Device_Instance.h"
#include "../Device/Device_Manager.h"

namespace py = pybind11;

///////////////////////////////////////////////////////////////////////
/// \addtogroup Vector
/// @{
namespace pysycl {
///////////////////////////////////////////////////////////////////////
/// \brief Vector class for PySYCL
template <typename Scalar_type> class Vector {
public:
  ///////////////////////////////////////////////////////////////////////
  /// \brief Defining the device scalar type.
  using Scalar_T = Scalar_type;

  using Vector_T = Vector<Scalar_T>;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Defining the device type.
  using Device_T = pysycl::Device_Instance;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Basic constructor that takes in the size and device of
  ///        the vector.
  /// \param[in] size_in Number of elements in the vector.
  /// \param[in] device_in The target sycl device (Optional).
  Vector(int size_in, Device_T &device_in = get_device())
      : size(size_in), data_host(size), device(device_in),
        Q(device_in.get_queue()) {
    if (size <= 0)
      throw std::runtime_error(
          "ERROR IN VECTOR: number of elements must be > 0.");
    data_device = sycl::malloc_device<Scalar_T>(size, Q);
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Constructor that takes in a numpy array.
  /// \param[in] np_array_in The input numpy array.
  /// \param[in] device_in The target sycl device (Optional).
  Vector(py::array_t<Scalar_T> np_array_in, Device_T &device_in = get_device())
      : device(device_in), Q(device_in.get_queue()) {
    if (np_array_in.ndim() != 1)
      throw std::runtime_error("The input numpy array must be 1D.");

    auto unchecked = np_array_in.template unchecked<1>();
    size = unchecked.shape(0);
    data_host.resize(size);

    if (size <= 0)
      throw std::runtime_error(
          "ERROR IN VECTOR: number of elements must be > 0.");

    for (int i = 0; i < size; ++i) {
      data_host[i] = unchecked(i);
    }

    data_device = sycl::malloc_device<Scalar_T>(size, Q);
    mem_to_gpu();
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Copy constructor.
  Vector(const Vector &og)
      : size(og.size), data_host(og.data_host), device(og.device), Q(og.Q) {
    data_device = sycl::malloc_device<Scalar_T>(size, Q);
    Q.memcpy(data_device, og.data_device, size * sizeof(Scalar_T)).wait();
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Move constructor.
  Vector(Vector &&og) noexcept
      : size(std::exchange(og.size, 0)), data_host(std::move(og.data_host)),
        data_device(std::exchange(og.data_device, nullptr)), Q(og.Q),
        device(og.device) {}

  ///////////////////////////////////////////////////////////////////////
  /// \brief Copy assignment operator.
  Vector &operator=(const Vector &og) {
    data_host = og.data_host;
    sycl::free(data_device, Q);
    size = og.size;
    device = og.device;
    Q = og.Q;
    data_device = sycl::malloc_device<Scalar_T>(size, Q);
    Q.memcpy(data_device, og.data_device, size * sizeof(Scalar_T)).wait();

    return *this;
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Move assignment operator.
  Vector &operator=(Vector &&og) noexcept {
    data_host = std::move(og.data_host);
    sycl::free(data_device, Q);
    data_device = std::exchange(og.data_device, nullptr);
    size = std::exchange(og.size, 0);
    Q = og.Q;
    device = og.device;

    return *this;
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Destructor.
  ~Vector() {
    if (data_device) {
      sycl::free(data_device, Q);
    }

    data_device = nullptr;
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Overloaded operator() direct element access
  Scalar_T &operator()(int i) {
    if (i < 0 || i >= size)
      throw std::out_of_range("Vector access out of range");
    return data_host[i];
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Overloaded operator() read-only element access
  const Scalar_T &operator()(int i) const {
    if (i < 0 || i >= size)
      throw std::out_of_range("Vector access out of range");
    return data_host[i];
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Overloaded operator+ for vector addition (creating a new vector).
  /// \param[in] B The Vector that is being added.
  /// \return Vector representing the sum of the addition.
  Vector operator+(const Vector &B) const {
    auto res = Vector(size, this->device);
    binary_vector_operations<BinaryOperation::ADD>(B, res);
    return res;
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Overloaded operator+= for vector addition (edits self).
  /// \param[in] B The Vector that is being added to self.
  Vector &operator+=(const Vector &B) {
    binary_vector_operations<BinaryOperation::ADD>(B, *this);
    return *this;
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Overloaded operator- for vector subtraction (creating a new
  /// vector). \param[in] B The Vector that is being subtracted. \return
  /// Vector representing the difference of the subtraction.
  Vector operator-(const Vector &B) const {
    auto res = Vector(size, this->device);
    binary_vector_operations<BinaryOperation::SUBTRACT>(B, res);
    return res;
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Overloaded operator-= for vector subtraction (edits self).
  /// \param[in] B The Vector that is being subtracted.
  Vector &operator-=(const Vector &B) {
    binary_vector_operations<BinaryOperation::SUBTRACT>(B, *this);
    return *this;
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Overloaded operator* for element-wise multiplication (creating a
  /// new vector). \param[in] B The Vector that is being multiplied. \return
  /// Vector representing the product of the multiplication.
  Vector operator*(const Vector &B) const {
    auto res = Vector(size, this->device);
    binary_vector_operations<BinaryOperation::MULTIPLY>(B, res);
    return res;
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Overloaded operator*= for element-wise multiplication (edits
  /// self). \param[in] B The Vector that is being multiplied.
  Vector &operator*=(const Vector &B) {
    binary_vector_operations<BinaryOperation::MULTIPLY>(B, *this);
    return *this;
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Overloaded operator/ for element-wise division (creating a new
  /// vector). \param[in] B The Vector that is being divided. \return Vector
  /// representing the result of the division.
  Vector operator/(const Vector &B) const {
    auto res = Vector(size, this->device);
    binary_vector_operations<BinaryOperation::DIVIDE>(B, res);
    return res;
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Overloaded operator/= for element-wise division (edits self).
  /// \param[in] B The Vector that is being divided.
  Vector &operator/=(const Vector &B) {
    binary_vector_operations<BinaryOperation::DIVIDE>(B, *this);
    return *this;
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Get a reference to the SYCL queue.
  /// \return Reference to the SYCL queue.
  auto& dev() { return device; }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Get the number of elements in the Vector.
  /// \return Number of elements in the Vector.
  int get_size() const { return size; }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Get the data pointer from Vector.
  /// \return Pointer to Vector host data.
  std::vector<Scalar_T> get_host_data_vector() { return data_host; }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Get the data pointer from Vector.
  /// \return Pointer to Vector device data.
  Scalar_T *get_device_data_ptr() const { return data_device; }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Get platform index for the sycl device
  int get_platform_index() { return device.get_platform_index(); }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Get platform index for the sycl device
  int get_device_index() { return device.get_device_index(); }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Copy memory from the CPU to the GPU.
  void mem_to_gpu() {
    Q.memcpy(data_device, data_host.data(), size * sizeof(Scalar_T)).wait();
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Copy memory from the GPU to the CPU
  void mem_to_cpu() {
    Q.memcpy(data_host.data(), data_device, size * sizeof(Scalar_T)).wait();
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Get the device from Vector.
  /// \return The Vector device instance.
  pysycl::Device_Instance &get_local_device() { return device; }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Fill the device with a specific value
  void fill(Scalar_T C) {
    const size_t B = device.get_max_workgroup_size();
    Q.submit([&](sycl::handler &h) {
       const size_t global_size = ((size + B - 1) / B) * B;
       const auto N = size;
       const auto A = data_device;

       sycl::range<1> global{global_size};
       sycl::range<1> local{B};

       h.parallel_for(sycl::nd_range<1>(global, local),
                      [=](sycl::nd_item<1> it) {
                        const auto i = it.get_global_id();

                        if (i >= N)
                          return;

                        A[i] = C;
                      });
     }).wait();
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Function that finds the maximum value in the vector
  /// \return Maximum value in the vector
  auto max() {
    return reductions(sycl::maximum<Scalar_T>(),
                      std::numeric_limits<Scalar_T>::lowest());
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Function that finds the minimum value in the vector
  /// \return Minimum value in the vector
  auto min() {
    return reductions(sycl::minimum<Scalar_T>(),
                      std::numeric_limits<Scalar_T>::max());
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Function that finds the sum of all value in the vector
  /// \return Sum of all values in the vector
  auto sum() { return reductions(sycl::plus<Scalar_T>()); }

private:
  ///////////////////////////////////////////////////////////////////////
  /// \brief Number of elements in the vector.
  int size;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Vector for data stored in host memory.
  std::vector<Scalar_T> data_host;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Pointer to data stored in device memory.
  Scalar_T *data_device;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Device that will store and handle Vector memory and operations
  Device_T &device;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Device SYCL queue.
  sycl::queue &Q;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Defining enumerations for binary operations
  enum class BinaryOperation { ADD, SUBTRACT, MULTIPLY, DIVIDE };

  ///////////////////////////////////////////////////////////////////////
  /// \brief Function to perform binary matrix operations
  template <BinaryOperation op>
  void binary_vector_operations(const Vector &B, Vector &C) const {
    const auto rows = this->get_size();

    if (size != B.get_size()) {
      throw std::runtime_error("ERROR: Incompatible Vector dimensions.");
    }

    const auto platform_idx = this->device.get_platform_index();
    const auto device_idx = this->device.get_device_index();

    if (platform_idx != B.device.get_platform_index() ||
        device_idx != B.device.get_device_index()) {
      throw std::runtime_error("ERROR: Incompatible PySYCL device.");
    }

    const size_t wg_size = this->device.get_max_workgroup_size();

    Q.submit([&](sycl::handler &h) {
       const size_t global_size = ((size + wg_size - 1) / wg_size) * wg_size;
       const auto N = size;

       sycl::range<1> global{global_size};
       sycl::range<1> local{wg_size};

       auto data_1 = this->get_device_data_ptr();
       auto data_2 = B.get_device_data_ptr();
       auto data_new = C.get_device_data_ptr();

       h.parallel_for(sycl::nd_range<1>(global, local),
                      [=](sycl::nd_item<1> it) {
                        const auto i = it.get_global_id();

                        if (i >= N)
                          return;

                        if constexpr (op == BinaryOperation::ADD) {
                          data_new[i] = data_1[i] + data_2[i];
                        } else if constexpr (op == BinaryOperation::SUBTRACT) {
                          data_new[i] = data_1[i] - data_2[i];
                        } else if constexpr (op == BinaryOperation::MULTIPLY) {
                          data_new[i] = data_1[i] * data_2[i];
                        } else if constexpr (op == BinaryOperation::DIVIDE) {
                          data_new[i] = data_1[i] / data_2[i];
                        }
                      });
     }).wait();
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Function to perform reduction operations
  template <typename Operation_T>
  auto reductions(Operation_T &&op, Scalar_T val = 0.0) {
    sycl::buffer<Scalar_T> buf{&val, 1};

    const size_t wg_size = device.get_max_workgroup_size();

    Q.submit([&](sycl::handler &h) {
       const auto reduction_func =
           sycl::reduction(buf, h, std::forward<Operation_T>(op));

       const size_t global_size = ((size + wg_size - 1) / wg_size) * wg_size;
       sycl::range<1> global{global_size};
       sycl::range<1> local{wg_size};

       auto data_device_ptr = data_device;
       const size_t N = size;

       h.parallel_for(sycl::nd_range<1>(global, local), reduction_func,
                      [=](sycl::nd_item<1> it, auto &el) {
                        const auto idx = it.get_global_id();
                        if (idx >= N)
                          return;

                        el.combine(data_device_ptr[idx]);
                      });
     }).wait();

    sycl::host_accessor val_host{buf, sycl::read_only};
    val = val_host[0];

    return val;
  }
}; // class Vector

/// @} // end "Vector" doxygen group

} // namespace pysycl

#endif // VECTOR_TYPE_H
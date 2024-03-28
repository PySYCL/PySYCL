#ifndef MATRIX_TYPE_H
#define MATRIX_TYPE_H

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
/// \brief Matrix Type Object in PySYCL.
///////////////////////////////////////////////////////////////////////

/// 2-Dimensional matrices in PySYCL

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
/// \addtogroup Matrix
/// @{
namespace pysycl {
///////////////////////////////////////////////////////////////////////
/// \brief Matrix class for PySYCL
template <typename Scalar_type> class Matrix {
public:
  ///////////////////////////////////////////////////////////////////////
  /// \brief Defining the device scalar type.
  using Scalar_T = Scalar_type;

  using Matrix_T = Matrix<Scalar_T>;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Defining the device type.
  using Device_T = pysycl::Device_Instance;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Basic constructor that takes in the size and device of
  ///        the matrix.
  /// \param[in] rows_in Number of rows in the matrix.
  /// \param[in] cols_in Number of columns in the matrix.
  /// \param[in] device_in Number of elements in the matrix (Optional).
  Matrix(int rows_in, int cols_in, Device_T &device_in = get_device())
      : rows(rows_in), cols(cols_in), data_host(rows * cols), device(device_in),
        Q(device_in.get_queue()) {
    if (rows <= 0 || cols <= 0)
      throw std::runtime_error(
          "ERROR IN MATRIX: number of cols and rows must be > 0.");

    data_device = sycl::malloc_device<Scalar_T>(rows * cols, Q);
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Constructor that takes in a numpy array.
  /// \param[in] np_array_in The input numpy array.
  /// \param[in] device_in The target device (Optional).
  Matrix(py::array_t<Scalar_T> np_array_in, Device_T &device_in = get_device())
      : device(device_in), Q(device_in.get_queue()) {
    if (np_array_in.ndim() != 2)
      throw std::runtime_error("The input numpy array must be 2D.");

    auto unchecked = np_array_in.template unchecked<2>();
    rows = unchecked.shape(0);
    cols = unchecked.shape(1);
    data_host.resize(rows * cols);

    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        data_host[i * cols + j] = unchecked(i, j);
      }
    }

    data_device = sycl::malloc_device<Scalar_T>(rows * cols, Q);
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Copy constructor.
  Matrix(const Matrix &og)
      : rows(og.rows), cols(og.cols), data_host(og.data_host),
        device(og.device), Q(og.Q) {
    data_device = sycl::malloc_device<Scalar_T>(rows * cols, Q);
    Q.memcpy(data_device, og.data_device, rows * cols * sizeof(Scalar_T))
        .wait();
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Move constructor.
  Matrix(Matrix &&og) noexcept
      : rows(std::exchange(og.rows, 0)), cols(std::exchange(og.cols, 0)),
        data_host(std::move(og.data_host)),
        data_device(std::exchange(og.data_device, nullptr)), Q(og.Q),
        device(og.device) {}

  ///////////////////////////////////////////////////////////////////////
  /// \brief Copy assignment operator.
  Matrix &operator=(const Matrix &og) {
    data_host = og.data_host;
    sycl::free(data_device, Q);
    rows = og.rows;
    cols = og.cols;
    device = og.device;
    Q = og.Q;
    data_device = sycl::malloc_device<Scalar_T>(rows * cols, Q);
    Q.memcpy(data_device, og.data_device, rows * cols * sizeof(Scalar_T))
        .wait();

    return *this;
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Move assignment operator.
  Matrix &operator=(Matrix &&og) noexcept {
    data_host = std::move(og.data_host);
    sycl::free(data_device, Q);
    data_device = std::exchange(og.data_device, nullptr);
    rows = std::exchange(og.rows, 0);
    cols = std::exchange(og.cols, 0);
    Q = og.Q;
    device = og.device;

    return *this;
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Destructor.
  ~Matrix() {
    if (data_device) {
      sycl::free(data_device, Q);
    }

    data_device = nullptr;
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Overloaded operator() direct element access
  Scalar_T &operator()(int i, int j) {
    if (i < 0 || i >= rows || j < 0 || j >= cols)
      throw std::out_of_range("Matrix access out of range");
    return data_host[i * cols + j];
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Overloaded operator() read-only element access
  const Scalar_T &operator()(int i, int j) const {
    if (i < 0 || i >= rows || j < 0 || j >= cols)
      throw std::out_of_range("Matrix access out of range");
    return data_host[i * cols + j];
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Overloaded operator+ for matrix addition (creating a new matrix).
  /// \param[in] B The Matrix that is being added.
  /// \return Matrix representing the sum of the addition.
  Matrix operator+(const Matrix &B) const {
    auto res = Matrix(rows, cols, this->device);
    binary_matrix_operations<BinaryOperation::ADD>(B, res);
    return res;
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Overloaded operator+= for matrix addition (edits self).
  /// \param[in] B The Matrix that is being added to self.
  Matrix &operator+=(Matrix &B) {
    binary_matrix_operations<BinaryOperation::ADD>(B, *this);
    return *this;
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Overloaded operator- for matrix subtraction (creating a new
  /// matrix).
  /// \param[in] B The Matrix that is being subtracted.
  /// \return Matrix representing the difference of the subtraction.
  Matrix operator-(Matrix &B) {
    auto res = Matrix(rows, cols, this->device);
    binary_matrix_operations<BinaryOperation::SUBTRACT>(B, res);
    return res;
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Overloaded operator-= for matrix subtraction (edits self).
  /// \param[in] B The Matrix that is being subtracted.
  Matrix &operator-=(Matrix &B) {
    binary_matrix_operations<BinaryOperation::SUBTRACT>(B, *this);
    return *this;
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Overloaded operator* for element-wise multiplication (creating a
  /// new matrix).
  /// \param[in] B The Matrix that is being multiplied.
  /// \return Matrix representing the product of the multiplication.
  Matrix operator*(Matrix &B) {
    auto res = Matrix(rows, cols, this->device);
    binary_matrix_operations<BinaryOperation::MULTIPLY>(B, res);
    return res;
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Overloaded operator*= for element-wise multiplication (edits
  /// self).
  /// \param[in] B The Matrix that is being multiplied.
  Matrix &operator*=(Matrix &B) {
    binary_matrix_operations<BinaryOperation::MULTIPLY>(B, *this);
    return *this;
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Overloaded operator/ for element-wise division (creating a new
  /// matrix).
  /// \param[in] B The Matrix that is being divided.
  /// \return Matrix representing the result of the division.
  Matrix operator/(Matrix &B) {
    auto res = Matrix(rows, cols, this->device);
    binary_matrix_operations<BinaryOperation::DIVIDE>(B, res);
    return res;
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Overloaded operator/= for element-wise division (edits self).
  /// \param[in] B The Matrix that is being divided.
  Matrix &operator/=(Matrix &B) {
    binary_matrix_operations<BinaryOperation::DIVIDE>(B, *this);
    return *this;
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Get a reference to the SYCL queue.
  /// \return Reference to the SYCL queue.
  auto& dev() { return device; }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Get the number of rows in the Matrix.
  /// \return Number of rows in the Matrix.
  int num_rows() const { return rows; }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Get the number of columns in the Matrix.
  /// \return Number of columns in the Matrix.
  int num_cols() const { return cols; }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Get the data pointer from Matrix.
  /// \return Pointer to Matrix host data.
  std::vector<Scalar_T> get_host_data_vector() { return data_host; }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Get the data pointer from Matrix.
  /// \return Pointer to Matrix device data.
  Scalar_T *get_device_data_ptr() const { return data_device; }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Copy memory from the CPU to the GPU.
  void mem_to_gpu() {
    Q.memcpy(data_device, &data_host[0], rows * cols * sizeof(Scalar_T)).wait();
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Copy memory from the GPU to the CPU
  void mem_to_cpu() {
    Q.memcpy(&data_host[0], data_device, rows * cols * sizeof(Scalar_T)).wait();
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Fill the device with a specific value
  void fill(const Scalar_T C) {
    const size_t B = sqrt(device.get_max_workgroup_size());

    Q.submit([&](sycl::handler &h) {
       const size_t global_size_rows = ((rows + B - 1) / B) * B;
       const size_t global_size_cols = ((cols + B - 1) / B) * B;
       const auto M = rows;
       const auto N = cols;
       const auto A = data_device;

       sycl::range<2> global{global_size_rows, global_size_cols};
       sycl::range<2> local{B, B};

       h.parallel_for(sycl::nd_range<2>(global, local),
                      [=](sycl::nd_item<2> it) {
                        const auto i = it.get_global_id(0);
                        const auto j = it.get_global_id(1);

                        if (i >= M || j >= N)
                          return;

                        A[i * N + j] = C;
                      });
     }).wait();
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Function that finds the maximum value in the matrix
  /// \return Maximum value in the matrix
  auto max() {
    return reductions(sycl::maximum<Scalar_T>(),
                      std::numeric_limits<Scalar_T>::lowest());
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Function that finds the minimum value in the matrix
  /// \return Minimum value in the matrix
  auto min() {
    return reductions(sycl::minimum<Scalar_T>(),
                      std::numeric_limits<Scalar_T>::max());
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Function that finds the sum of all values in the matrix
  /// \return Sum of all values in the matrix
  auto sum() { return reductions(sycl::plus<Scalar_T>()); }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Function that transposes the matrix
  void transpose() {
    const int rows_old = rows;
    const int cols_old = cols;

    rows = cols_old;
    cols = rows_old;

    Scalar_T* old_data = sycl::malloc_device<Scalar_T>(rows_old * cols_old, Q);

    auto setup = Q.submit([&](sycl::handler &h) {
      auto data_device_ptr = data_device;

      h.parallel_for(sycl::range<2>(rows_old, cols_old), [=](sycl::id<2> idx){
        const int i = idx[0];
        const int j = idx[1];

        old_data[i * cols_old + j] = data_device_ptr[i * cols_old + j];
      });
    });

    Q.submit([&](sycl::handler &h) {
      auto data_device_ptr = data_device;

      h.depends_on(setup);

      h.parallel_for(sycl::range<2>(rows_old, cols_old), [=](sycl::id<2> idx){
        const int i = idx[0];
        const int j = idx[1];

        data_device_ptr[j * rows_old + i] = old_data[i * cols_old + j];
      });
    }).wait();

    sycl::free(old_data, Q);
  }

private:
  ///////////////////////////////////////////////////////////////////////
  /// \brief Number of rows in the matrix.
  int rows;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Number of columns in the matrix.
  int cols;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Vector for data stored in host memory.
  std::vector<Scalar_T> data_host;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Pointer to data stored in device memory.
  Scalar_T *data_device;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Device that will store and handle Matrix memory and operations
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
  void binary_matrix_operations(const Matrix &B, Matrix &C) const {
    const auto rows = this->num_rows();
    const auto cols = this->num_cols();

    if (rows != B.num_rows() || cols != B.num_cols()) {
      throw std::runtime_error("ERROR: Incompatible Matrix dimensions.");
    }

    const auto platform_idx = this->device.get_platform_index();
    const auto device_idx = this->device.get_device_index();

    if (platform_idx != B.device.get_platform_index() ||
        device_idx != B.device.get_device_index()) {
      throw std::runtime_error("ERROR: Incompatible PySYCL device.");
    }

    const size_t wg_size = sqrt(this->device.get_max_workgroup_size());

    Q.submit([&](sycl::handler &h) {
       const size_t global_size_rows =
           ((rows + wg_size - 1) / wg_size) * wg_size;
       const size_t global_size_cols =
           ((cols + wg_size - 1) / wg_size) * wg_size;

       sycl::range<2> global{global_size_rows, global_size_cols};
       sycl::range<2> local{wg_size, wg_size};

       auto data_1 = this->get_device_data_ptr();
       auto data_2 = B.get_device_data_ptr();
       auto data_new = C.get_device_data_ptr();

       h.parallel_for(sycl::nd_range<2>(global, local),
                      [=](sycl::nd_item<2> it) {
                        const auto i = it.get_global_id(0);
                        const auto j = it.get_global_id(1);

                        if (i >= rows || j >= cols)
                          return;

                        if constexpr (op == BinaryOperation::ADD) {
                          data_new[i * cols + j] =
                              data_1[i * cols + j] + data_2[i * cols + j];
                        } else if constexpr (op == BinaryOperation::SUBTRACT) {
                          data_new[i * cols + j] =
                              data_1[i * cols + j] - data_2[i * cols + j];
                        } else if constexpr (op == BinaryOperation::MULTIPLY) {
                          data_new[i * cols + j] =
                              data_1[i * cols + j] * data_2[i * cols + j];
                        } else if constexpr (op == BinaryOperation::DIVIDE) {
                          data_new[i * cols + j] =
                              data_1[i * cols + j] / data_2[i * cols + j];
                        }
                      });
     }).wait();
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Function to perform reduction operations
  template <typename Operation_T>
  auto reductions(Operation_T op, Scalar_T val = 0.0) {
    sycl::buffer<Scalar_T> buf{&val, 1};

    const size_t wg_size = device.get_max_workgroup_size();

    Q.submit([&](sycl::handler &h) {
       const auto reduction_func =
           sycl::reduction(buf, h, std::forward<Operation_T>(op));

       const size_t global_size =
           ((rows * cols + wg_size - 1) / wg_size) * wg_size;
       sycl::range<1> global{global_size};
       sycl::range<1> local{wg_size};

       auto data_device_ptr = data_device;
       const size_t N = rows * cols;

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
}; // class Matrix

/// @} // end "Matrix" doxygen group

} // namespace pysycl

#endif // MATRIX_TYPE_H
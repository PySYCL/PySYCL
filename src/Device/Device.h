#ifndef DEVICE_H
#define DEVICE_H

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
/// \brief Device object in PySYCL.
///////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////
// stl
///////////////////////////////////////////////////////////////////////
#include <stdexcept>

///////////////////////////////////////////////////////////////////////
// sycl
///////////////////////////////////////////////////////////////////////
#include <sycl/sycl.hpp>

///////////////////////////////////////////////////////////////////////
/// \addtogroup Device
/// @{

namespace pysycl {

///////////////////////////////////////////////////////////////////////
/// \brief Class defining a device for device selection.
class Device {
  public:
  /////////////////////////////////////////////////////////////////////
  /// \brief Copy constructor, use compiler generated version.
  Device(const Device&) = default;

  /////////////////////////////////////////////////////////////////////
  /// \brief Move constructor, use compiler generated version.
  Device(Device&&) = default;

  /////////////////////////////////////////////////////////////////////
  /// \brief Copy assignment, use compiler generated version.
  /// \return reference to the assigned object.
  Device& operator=(const Device&) = default;

  /////////////////////////////////////////////////////////////////////
  /// \brief Move assignment, use compiler generated version.
  /// \return reference to the assigned object.
  Device& operator=(Device&&) = default;

  /////////////////////////////////////////////////////////////////////
  /// \brief The platform index.
  int platform_idx;

  /////////////////////////////////////////////////////////////////////
  /// \brief The device index based on the platform index.
  int device_idx;

  /////////////////////////////////////////////////////////////////////
  /// \brief Output device name.
  auto name() { return Q.get_device().get_info<sycl::info::device::name>(); }

  /////////////////////////////////////////////////////////////////////
  /// \brief Output device vendor.
  auto vendor() {
    return Q.get_device().get_info<sycl::info::device::vendor>();
  }

  /////////////////////////////////////////////////////////////////////
  /// \brief Returns the device queue.
  auto& get_queue() { return Q; }

  /////////////////////////////////////////////////////////////////////
  /// \brief Returns the platform index.
  auto get_platform_index() const { return platform_idx; }

  /////////////////////////////////////////////////////////////////////
  /// \brief Returns the device index.
  auto get_device_index() const { return device_idx; }

  /////////////////////////////////////////////////////////////////////
  /// \brief Returns the maximum workgroup size of the device.
  auto get_max_workgroup_size() const {
    return Q.get_device().get_info<sycl::info::device::max_work_group_size>();
  }

  /////////////////////////////////////////////////////////////////////
  /// \brief Constructor that selects a SYCL device.
  /// \param[in] platform_index_in Index of the sycl platform to select.
  /// \param[in] device_index_in Index of the sycl device to select.
  Device(const int platform_idx_in = 0, const int device_idx_in = 0)
    : platform_idx(platform_idx_in)
    , device_idx(device_idx_in) {
    if (platform_idx < 0) {
      throw std::runtime_error("ERROR: Platform index must be non-negative.");
    }

    if (device_idx < 0) {
      throw std::runtime_error("ERROR: Device index must be non-negative.");
    }

    auto platforms = sycl::platform::get_platforms();

    if (platform_idx >= platforms.size()) {
      throw std::runtime_error("ERROR: Platform index out of range.");
    }

    auto devices = platforms[platform_idx].get_devices();

    if (device_idx >= devices.size()) {
      throw std::runtime_error("ERROR: Device index out of range.");
    }

    Q = sycl::queue(devices[device_idx]);
  }

  private:
  /////////////////////////////////////////////////////////////////////
  /// \brief The selected device queue.
  sycl::queue Q;
};

///////////////////////////////////////////////////////////////////////
// Overload operators.

///////////////////////////////////////////////////////////////////////
// Equality operator.
inline bool operator==(const Device& di1, const Device& di2) {
  return (di1.platform_idx == di2.platform_idx)
         && (di1.device_idx == di2.device_idx);
}

///////////////////////////////////////////////////////////////////////
// Inequality operator.
inline bool operator!=(const Device& di1, const Device& di2) {
  return !(di1 == di2);
}

///////////////////////////////////////////////////////////////////////
// Less than operator.
inline bool operator<(const Device& di1, const Device& di2) {
  if (di1.platform_idx != di2.platform_idx) {
    return di1.platform_idx < di2.platform_idx;
  } else {
    return di1.device_idx < di2.device_idx;
  }
}

///////////////////////////////////////////////////////////////////////
// Greater than operator.
inline bool operator>(const Device& di1, const Device& di2) {
  return !(di1 < di2 || di1 == di2);
}

///////////////////////////////////////////////////////////////////////
// Less than or equal to operator.
inline bool operator<=(const Device& di1, const Device& di2) {
  return (di1 < di2) || (di1 == di2);
}

///////////////////////////////////////////////////////////////////////
// Greater than or equal to operator.
inline bool operator>=(const Device& di1, const Device& di2) {
  return (di1 > di2) || (di1 == di2);
}

} // namespace pysycl

#endif // #ifndef DEVICE_H
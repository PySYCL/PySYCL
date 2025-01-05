# ######################################################################
# ######################################################################
# # CMakeLists.txt for PySYCL
# # subdirectory : Device
# ######################################################################
# ######################################################################
# # This file is part of the PySYCL software for SYCL development in
# # Python. It is licensed under the Apache License, Version 2.0. A copy
# # of this license, in a file named LICENSE.md, should have been
# # distributed with this file. A copy of this license is also
# # currently available at "http://www.apache.org/licenses/LICENSE-2.0".
# #
# # Unless explicitly stated, all contributions intentionally submitted
# # to this project shall also be under the terms and conditions of this
# # license, without any additional terms or conditions.
# ######################################################################
# ######################################################################

# ######################################################################
# # Tests
# ######################################################################
option(PYSYCL_TEST_DEVICE_INQUIRY "Compile tests for Device Inquiry" ON)
if(PYSYCL_TEST_DEVICE_INQUIRY)
  PySYCL_add_to_tests("Device_Inquiry_Tests.cpp")
endif()

# ######################################################################
# # Setting the name of the PyBind module
# ######################################################################
option(PYSYCL_Device "Compile the PySYCL device module" ON)

# ######################################################################
# # Python Module
# ######################################################################
if (PYSYCL_Device)
  PySYCL_add_pybind11_module(device Device_Python_Module.cpp)
endif()
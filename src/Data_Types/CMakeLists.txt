# ######################################################################
# ######################################################################
# # CMakeLists.txt for PySYCL
# # subdirectory : Data_Types
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
option(PYSYCL_TEST_DATA_TYPES "Compile tests for Data types" ON)
if(PYSYCL_TEST_DATA_TYPES)
  PySYCL_add_to_tests("Data_Types_Tests.cpp")
endif()

# ######################################################################
# # Setting the name of the PyBind module
# ######################################################################
option(PYSYCL_Data_Types "Compile the PySYCL data types module" ON)

# ######################################################################
# # Python Module
# ######################################################################
if (PYSYCL_Data_Types)
  PySYCL_add_pybind11_module(data_types Data_Types_Python_Module.cpp)
endif()
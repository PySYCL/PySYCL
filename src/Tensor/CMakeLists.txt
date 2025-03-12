# ######################################################################
# ######################################################################
# # CMakeLists.txt for PySYCL
# # subdirectory : Tensor
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
option(PYSYCL_TEST_TENSOR "Compile tests for Tensor" ON)
if(PYSYCL_TEST_TENSOR)
  PySYCL_add_to_tests("Tensor_Tests.cpp")
endif()

# ######################################################################
# # Setting the name of the PyBind module
# ######################################################################
option(PYSYCL_Tensor "Compile the PySYCL tensor module" ON)

# ######################################################################
# # Python Module
# ######################################################################
if (PYSYCL_Tensor)
  PySYCL_add_pybind11_module(tensor_types Tensor_Types_Python_Module.cpp)
endif()

# ######################################################################
# # Python Module
# ######################################################################
if (PYSYCL_Tensor)
  PySYCL_add_pybind11_module(tensor Tensor_Python_Module.cpp)
endif()
# ######################################################################
# ######################################################################
# #                  CMakeLists.txt for PySYCL
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

cmake_minimum_required(VERSION 3.13)
project(PySYCL)

# ######################################################################
# # Version of C++
# ######################################################################
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#######################################################################
## SYCL Backends
#######################################################################
option(PYSYCL_USE_CUDA "Use CUDA" OFF)

#######################################################################
## SYCL Extensions
#######################################################################
option(PYSYCL_USE_ONEMKL "Use OneMKL" OFF)

# ######################################################################
# # Include dependencies for each module
# ######################################################################
add_library(pysycl_project_dependencies INTERFACE)

# ######################################################################
# # Build types
# ######################################################################
include(CMake/BuildType.cmake)

# ######################################################################
# # Google test
# ######################################################################
include(CMake/GTest.cmake)

# ######################################################################
# # PyBind
# ######################################################################
include(CMake/PyBind11.cmake)

# ######################################################################
# # Subdirectories
# ######################################################################
add_subdirectory(Data_Types)
add_subdirectory(Device)
add_subdirectory(Linear_Algebra)
add_subdirectory(Tensor)
add_subdirectory(Utilities)
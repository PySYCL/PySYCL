# ######################################################################
# ######################################################################
# #                  CMakeLists.txt for PySYCL
# ######################################################################
# ######################################################################
# # This file is part of the PySYCL software for SYCL development in
# # Python.  It is licensed under the MIT licence.  A copy of
# # this license, in a file named LICENSE.md, should have been
# # distributed with this file.  A copy of this license is also
# # currently available at "http://opensource.org/licenses/MIT".
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
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ######################################################################
# # Build types
# ######################################################################
include(CMake/BuildType.cmake)

# ######################################################################
# # Include dependencies for each module
# ######################################################################
add_library(pysycl_project_dependencies INTERFACE)

# ######################################################################
# # PyBind
# ######################################################################
include(CMake/PyBind11.cmake)

# ######################################################################
# # Subdirectories
# ######################################################################
add_subdirectory(Data_Types)
add_subdirectory(Device)
add_subdirectory(Fast_Fourier_Transform)
add_subdirectory(General)
add_subdirectory(Linear_Algebra)
add_subdirectory(Matrix)
add_subdirectory(Vector)

# ######################################################################
# # Python Test Suite
# ######################################################################
include(CMake/PythonUnitTest.cmake)

# ######################################################################
# # Doxygen documentation
# ######################################################################
include(CMake/Doxygen.cmake)

# ######################################################################
# # Sphinx documentation
# ######################################################################
include(CMake/Sphinx.cmake)
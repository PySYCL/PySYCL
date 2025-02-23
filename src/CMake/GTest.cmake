# ######################################################################
# ######################################################################
# #                  src/CMake/GTest.cmake
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

#######################################################################
## Find package for Google unit tests
#######################################################################
find_package(GTest CONFIG REQUIRED)

#######################################################################
## Test suite application
#######################################################################
add_executable(Chunin_Exams test_suite.cpp)
target_link_libraries(Chunin_Exams PRIVATE GTest::gtest)
target_link_libraries(Chunin_Exams PRIVATE pysycl_project_dependencies)

function(PySYCL_add_to_tests file)
  target_sources(Chunin_Exams PRIVATE ${file})
endfunction()

target_compile_definitions(Chunin_Exams PRIVATE PYSYCL_USE_ONEMKL)

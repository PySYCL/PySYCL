# ######################################################################
# ######################################################################
# #                  src/CMake/Doxygen.cmake
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

set(PYSYCL_DOC_DIR "PySYCL_doxygen_html"
  CACHE STRING "Name of the documentation directory")

if("$PYSYCL_DOC_DIR}" STREQUAL "")
  message(FATAL_ERROR "PYSYCL_DOC_DIR cannot be blank.")
endif()

add_custom_target(doxygen COMMAND env DOXYGEN_OUTPUT_DIRECTORY=${CMAKE_BINARY_DIR}/../docs/
  env DOXYGEN_HTML_OUTPUT_DIRECTORY=${PYSYCL_DOC_DIR}
  doxygen Doxyfile > ${CMAKE_BINARY_DIR}/doxygen.log 2> ${CMAKE_BINARY_DIR}/doxygen.err
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/../doxygen/
  COMMENT
  "Build Doxygen documentation.
     HTML:     ${CMAKE_BINARY_DIR}/../docs/${PYSYCL_DOC_DIR}
     Output:   ${CMAKE_BINARY_DIR}/doxygen.log
     Warnings: ${CMAKE_BINARY_DIR}/doxygen.err")
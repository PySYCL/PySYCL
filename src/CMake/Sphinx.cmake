# ######################################################################
# ######################################################################
# #                  src/CMake/Sphinx.cmake
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
FIND_PROGRAM(SPHINX_API_EXE NAMES sphinx-apidoc)
FIND_PROGRAM(SPHINX_BUILD_EXE NAMES sphinx-build)

set(SPHINX_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../sphinx")
set(SPHINX_BUILD "${CMAKE_CURRENT_SOURCE_DIR}/../docs/PySYCL_sphinx_html")
set(SPHINX_MODULE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../build/pysycl")

add_custom_target(sphinx
  COMMAND ${SPHINX_API_EXE} -o ${SPHINX_SOURCE} ${SPHINX_MODULE_DIR}
  COMMAND ${SPHINX_BUILD_EXE} -b html ${SPHINX_SOURCE} ${SPHINX_BUILD}
  COMMENT "Generating Sphinx documentation"
)
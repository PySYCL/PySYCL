# ######################################################################
# ######################################################################
# #                  src/CMake/BuildType.cmake
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
# # Build types
# ######################################################################
set(DEFAULT_BUILD_TYPE "Release")

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  set(COMILER_ARCH_FLAG "-xHost")
else()
  set(COMPILER_ARCH_FLAG "-march=native")
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}', as none was specified.")
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE
    STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# ######################################################################
# # Flags for different backends
# ######################################################################
if(PYSYCL_USE_CUDA)
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG ${COMPILER_ARCH_FLAG} -fsycl -fsycl-targets=nvptx64-nvidia-cuda" 
  CACHE STRING "Release build flags." FORCE)

  set(CMAKE_CXX_FLAGS_DEBUG "-g -fsycl -fsycl-targets=nvptx64-nvidia-cuda"
  CACHE STRING "Debug build flags." FORCE)

  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELEASE} -g -fsycl -fsycl-targets=nvptx64-nvidia-cuda"
  CACHE STRING "Release build with debug info flags." FORCE)

  set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os -DNDEBUG -fsycl -fsycl-targets=nvptx64-nvidia-cuda ${COMPILER_ARCH_FLAG}"
  CACHE STRING "Min-size release build flags." FORCE)
else()
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG ${COMPILER_ARCH_FLAG} -fsycl" CACHE STRING
  "Release build flags." FORCE)

  set(CMAKE_CXX_FLAGS_DEBUG "-g -fsycl" CACHE STRING "Debug build flags." FORCE)

  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELEASE} -g -fsycl" CACHE STRING
  "Release build with debug info flags." FORCE)

  set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os -DNDEBUG -fsycl ${COMPILER_ARCH_FLAG}" CACHE STRING
  "Min-size release build flags." FORCE)
endif()
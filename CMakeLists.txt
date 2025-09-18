# __________________________________________________________________
#
# CMakeList for Mini-PIC
# 
# __________________________________________________________________

cmake_minimum_required (VERSION 3.5)

message(" __________________________________________________________________ \n")
message(" CMakeList for Mini-PIC")
message(" __________________________________________________________________ \n")

# Disable in-source builds to prevent source tree corruption.
if( "${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}" )
 message( FATAL_ERROR "FATAL: In-source builds are not allowed. You should create a separate directory for build files and delete CMakeCache.txt." )
endif()

# __________________________________________________________________
# Options

# Debug, Release, RelWithDebInfo and MinSizeRel
#set(CMAKE_BUILD_TYPE Release)
message(" Build type: ${CMAKE_BUILD_TYPE}")

# Backend
set(BACKEND "sequential" CACHE STRING "Backend to use (sequential (default), openmp, kokkos)")

# Atomic type for projection using Kokkos
set(KOKKOS_PROJECTION_TYPE "scatter_view" CACHE STRING "Type of projection for Kokkos (scatter_view (default), atomic_view)")

if (BACKEND STREQUAL "kokkos")
  message(" Kokkos projection type: ${KOKKOS_PROJECTION_TYPE}")
endif()

add_definitions(-DBACKEND="${BACKEND}")

# Debug and test modes
option(DEBUG "Debug messages" OFF)
option(WARNING "Warning mode (ON/OFF)" OFF)
option(TEST "Simulation test for CI" OFF)
option(UNIT_TESTS "Compile Unit tests" OFF)
option(MINIPIC "Compile minipic (ON/OFF)" ON)


message(" Minipic compilation: ${MINIPIC}")
message(" Unit tests: ${UNIT_TESTS}")

# Check that the backend is supported
if (BACKEND STREQUAL "sequential")
  message(" Backend: sequential")
  set(EXE_NAME minipic.seq)
elseif (BACKEND STREQUAL "openmp")
  message(" Backend: openmp")
  set(EXE_NAME minipic.omp)
elseif (BACKEND STREQUAL "kokkos")
  message(" Backend: kokkos")
  set(EXE_NAME minipic.kokkos)
elseif (BACKEND STREQUAL "kokkos_dualview_unified")
  message(" Backend: kokkos using dual views with unified memory")
  set(EXE_NAME minipic.kokkos_dualview_unified)
elseif (BACKEND STREQUAL "kokkos_unified")
  message(" Backend: kokkos with unified memory")
  set(EXE_NAME minipic.kokkos_unified)
else()
  message(FATAL_ERROR "FATAL: Backend (${BACKEND}) not supported")
endif()

message(" Executable name: ${EXE_NAME}")

# Specific architecture option
set(DEVICE "auto" CACHE STRING "Device architecture (auto (default), nvidia_v100)")

# Check that the architecture is supported
if ((DEVICE STREQUAL "auto") OR (DEVICE STREQUAL "nvidia_v100") OR (DEVICE STREQUAL "nvidia_a100") OR (DEVICE STREQUAL "amd_mi250") OR (DEVICE STREQUAL "nvidia_gh200") OR (DEVICE STREQUAL "nvidia_h100") OR (DEVICE STREQUAL "intel_pvc") OR (DEVICE STREQUAL "amd_mi300") OR (DEVICE STREQUAL "amd_genoa") OR (DEVICE STREQUAL "nvidia_grace"))
 message(" Device architecture: ${DEVICE}")
else()
 message(FATAL_ERROR "FATAL: Device (${DEVICE}) architecture not supported")
endif()

# _________________________________________________________________
# Debug mode

if (DEBUG)
 message(" Debug mode: ON")
endif()

# _________________________________________________________________
# Warning mode

if (WARNING)
 message(" Warning mode: ON")
 set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
endif()

# __________________________________________________________________
# Determine languages

set(LANGUAGES CXX)

if ((BACKEND STREQUAL "thrust") OR (BACKEND STREQUAL "stdpar") OR (BACKEND STREQUAL "thrust_unified"))
 if ((DEVICE STREQUAL "nvidia_v100") OR (DEVICE STREQUAL "nvidia_a100") OR (DEVICE STREQUAL "nvidia_h100") OR (DEVICE STREQUAL "auto") OR (DEVICE STREQUAL "nvidia_gh200"))
 set(LANGUAGES ${LANGUAGES} CUDA)
 endif()
endif()

message(" Languages: ${LANGUAGES}")

message(" __________________________________________________________________ \n")

project (minipic LANGUAGES ${LANGUAGES})

# __________________________________________________________________

# Compiler
#set(CMAKE_CXX_COMPILER clang++)
#set(CMAKE_CXX_COMPILER icpx)
#set(CMAKE_CXX_COMPILER icpc)
#set(CMAKE_CXX_COMPILER g++)

# message("${CMAKE_CURRENT_SOURCE_DIR}")
# message("${CMAKE_SOURCE_DIR}")

# Specific behavior for test mode
if (TEST)
  set(MAIN_PATH "./main.cpp")
else()
 set(MAIN_PATH "./src/main.cpp")
endif()

set(COMPILE_OPTIONS "")
set(LINK_OPTIONS "")

# __________________________________________________________________
# Sources and compile options

# -----> OpenMP mode <------
if (BACKEND STREQUAL "openmp")

 set(COMPILE_OPTIONS ${COMPILE_OPTIONS} -fopenmp)
 set(COMPILE_OPTIONS ${COMPILE_OPTIONS} -std=c++17)
 set(COMPILE_OPTIONS ${COMPILE_OPTIONS} -D__MINIPIC_OMP__ -D__MINIPIC_SIMD__)
 set(COMPILE_OPTIONS ${COMPILE_OPTIONS} ${ARCH_OPT_FLAGS})

 include_directories(src/setups src/openmp src/common)

 # list of source files - automatic search cpp files in src/openmp and src/common directory
 file(GLOB_RECURSE SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/openmp/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/common/*.cpp")

 find_package(OpenMP REQUIRED)

# -----> Kokkos mode <------
elseif (BACKEND STREQUAL "kokkos")

 set(COMPILE_OPTIONS ${COMPILE_OPTIONS} -std=c++17 -D__MINIPIC_KOKKOS__ -D__MINIPIC_KOKKOS_COMMON__ -D__MINIPIC_KOKKOS_DUALVIEW_COMMON__ ${ARCH_OPT_FLAGS})

 if (KOKKOS_PROJECTION_TYPE STREQUAL "scatter_view")
  set(COMPILE_OPTIONS ${COMPILE_OPTIONS} -D__MINIPIC_KOKKOS_SCATTERVIEW__)
 else() #if (KOKKOS_PROJECTION_TYPE STREQUAL "atomic_view")
  set(COMPILE_OPTIONS ${COMPILE_OPTIONS} -D__MINIPIC_KOKKOS_ATOMICVIEW__)
 endif()

 include_directories(src/setups src/kokkos src/common)

 # list of source files - automatic search cpp files in src/openmp and src/common directory
 file(GLOB_RECURSE SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/kokkos/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/common/*.cpp")

 find_package(Kokkos REQUIRED)

# -----> Kokkos mode using unified memory <------
elseif (BACKEND STREQUAL "kokkos_dualview_unified")

 set(COMPILE_OPTIONS ${COMPILE_OPTIONS} -std=c++17 -D__MINIPIC_KOKKOS_DUALVIEW_UNIFIED__ -D__MINIPIC_KOKKOS_COMMON__ -D__MINIPIC_KOKKOS_DUALVIEW_COMMON__ ${ARCH_OPT_FLAGS})

 if (KOKKOS_PROJECTION_TYPE STREQUAL "scatter_view")
  set(COMPILE_OPTIONS ${COMPILE_OPTIONS} -D__MINIPIC_KOKKOS_SCATTERVIEW__)
 else() #if (KOKKOS_PROJECTION_TYPE STREQUAL "atomic_view")
  set(COMPILE_OPTIONS ${COMPILE_OPTIONS} -D__MINIPIC_KOKKOS_ATOMICVIEW__)
 endif()

 include_directories(src/setups src/kokkos src/common)

 # list of source files - automatic search cpp files in src/openmp and src/common directory
 file(GLOB_RECURSE SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/kokkos/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/common/*.cpp")

 find_package(Kokkos REQUIRED)

# -----> Kokkos mode using unified memory <------
elseif (BACKEND STREQUAL "kokkos_unified")

 set(COMPILE_OPTIONS ${COMPILE_OPTIONS} -std=c++17 -D__MINIPIC_KOKKOS_UNIFIED__ -D__MINIPIC_KOKKOS_COMMON__ ${ARCH_OPT_FLAGS})

 if (KOKKOS_PROJECTION_TYPE STREQUAL "scatter_view")
  set(COMPILE_OPTIONS ${COMPILE_OPTIONS} -D__MINIPIC_KOKKOS_SCATTERVIEW__)
 else() #if (KOKKOS_PROJECTION_TYPE STREQUAL "atomic_view")
  set(COMPILE_OPTIONS ${COMPILE_OPTIONS} -D__MINIPIC_KOKKOS_ATOMICVIEW__)
 endif()

 set(LINK_OPTIONS ${LINK_OPTIONS} ${ARCH_OPT_FLAGS})
 include_directories(src/setups src/kokkos src/common)

 # list of source files - automatic search cpp files in src/openmp and src/common directory
 file(GLOB_RECURSE SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/kokkos/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/common/*.cpp")

 find_package(Kokkos REQUIRED)

# ----> Serial mode <----
else()

 set(COMPILE_OPTIONS ${COMPILE_OPTIONS} -std=c++17 -D__MINIPIC_SIMD__)

 include_directories(src/setups src/openmp src/common)

 # list of source files - automatic search cpp files in src/openmp and src/common directory
 file(GLOB_RECURSE SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/openmp/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/common/*.cpp")

endif()

# -----> Debug mode <------
if (DEBUG)
 set(COMPILE_OPTIONS ${COMPILE_OPTIONS} -D__MINIPIC_DEBUG__)
endif()

# __________________________________________________________________
# Create executable with options

if (MINIPIC)

  add_executable(${EXE_NAME} ${SOURCES} ${MAIN_PATH})
  target_compile_options(${EXE_NAME} PRIVATE ${COMPILE_OPTIONS})
  target_link_options(${EXE_NAME} PRIVATE ${LINK_OPTIONS})

  # target compile options using EXE_NAME

  # ---->  OpenMP mode <----
  if (BACKEND STREQUAL "openmp")

    target_link_libraries("${EXE_NAME}" OpenMP::OpenMP_CXX)

  # ---->  Kokkos mode <----
  elseif (BACKEND STREQUAL "kokkos" OR BACKEND STREQUAL "kokkos_unified" OR BACKEND STREQUAL "kokkos_dualview_unified")
    target_link_libraries("${EXE_NAME}" Kokkos::kokkos)

  else()
    # nothing to do
  endif()

endif()

# tests

if (UNIT_TESTS)

 add_subdirectory(unit_tests/)

endif()



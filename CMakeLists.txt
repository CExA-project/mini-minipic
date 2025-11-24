# __________________________________________________________________
#
# CMakeList for Mini-PIC
# 
# __________________________________________________________________

cmake_minimum_required (VERSION 3.16)

message("------------------------------------------------------------------")
message("CMakeList for Mini-PIC")
message("------------------------------------------------------------------")

# Disable in-source builds to prevent source tree corruption.
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
  message(
    FATAL_ERROR
    "In-source builds are not allowed. You should create a separate directory "
    "for build files and delete CMakeCache.txt."
  )
endif()

# __________________________________________________________________
# Options

if(NOT CMAKE_BUILD_TYPE)
  set(
    CMAKE_BUILD_TYPE
    "RelWithDebInfo"
    CACHE
    STRING
    "Choose the type of build, options are: Debug, Release, RelWithDebInfo and MinSizeRel."
    FORCE
  )
endif()
message("Build type: ${CMAKE_BUILD_TYPE}")

# Debug and test modes
option(MINI_MINIPIC_DEBUG "Debug messages" OFF)
option(MINI_MINIPIC_WARNING "Warning mode" OFF)
option(MINI_MINIPIC_KOKKOS_SCATTER_VIEW "Use scatter views (Kokkos implementation)" OFF)
set(
  MINI_MINIPIC_IMPLEMENTATION
  "exercise"
  CACHE
  STRING
  "Which implementation to use (must be an existing directory in src/)"
)
set(
  MINI_MINIPIC_SETUP
  "antenna"
  CACHE
  STRING
  "Which setup to build and run with (must be the stem name of a valid .cpp file in src/setups/)"
)
set(
  MINI_MINIPIC_KOKKOS_SOURCE_DIRECTORY
  "${CMAKE_CURRENT_SOURCE_DIR}/external/kokkos"
  CACHE
  PATH
  "Path to the local source directory of Kokkos"
)

# _________________________________________________________________
# Debug mode

message("Debug mode: ${MINI_MINIPIC_DEBUG}")

# _________________________________________________________________
# Warning mode

if(MINI_MINIPIC_WARNING)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
endif()
message("Warning mode: ${MINI_MINIPIC_WARNING}")

# _________________________________________________________________
# Scatter view

message("Kokkos scatter view: ${MINI_MINIPIC_KOKKOS_SCATTER_VIEW}")

# _________________________________________________________________
# Implementation

set(IMPLEMENTATION_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/${MINI_MINIPIC_IMPLEMENTATION}")
if(NOT EXISTS "${IMPLEMENTATION_DIRECTORY}")
  message(FATAL_ERROR "Cannot find implementation '${MINI_MINIPIC_IMPLEMENTATION}' in src/")
endif()
message("Implementation: ${MINI_MINIPIC_IMPLEMENTATION}")

# _________________________________________________________________
# Setup

set(SETUP_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/setups/${MINI_MINIPIC_SETUP}.cpp")
if(NOT EXISTS "${SETUP_FILE}")
  message(FATAL_ERROR "Cannot find setup file ${MINI_MINIPIC_SETUP}.cpp in src/setups/")
endif()
message("Setup: ${MINI_MINIPIC_SETUP}")

message("------------------------------------------------------------------")

# __________________________________________________________________
# Project

project(
  minipic
  LANGUAGES CXX
)

# __________________________________________________________________
# Dependencies

find_package(Kokkos 5.0.0 QUIET)
if(Kokkos_FOUND)
  message(STATUS "Kokkos provided as installed: ${Kokkos_DIR} (version \"${Kokkos_VERSION}\")")
  message(STATUS "  With backend: ${Kokkos_DEVICES}")
  message(STATUS "  With options: ${Kokkos_OPTIONS}")
  message(STATUS "  For architectures: ${Kokkos_ARCH}")
elseif(EXISTS "${MINI_MINIPIC_KOKKOS_SOURCE_DIRECTORY}/CMakeLists.txt")
  message(STATUS "Kokkos provided as a directory: ${MINI_MINIPIC_KOKKOS_SOURCE_DIRECTORY}")
  add_subdirectory(external/kokkos)
else()
  message(STATUS "Kokkos downloaded: ${MINI_MINIPIC_KOKKOS_SOURCE_DIRECTORY}")
  include(FetchContent)
  FetchContent_Declare(
    kokkos
    DOWNLOAD_EXTRACT_TIMESTAMP ON
    URL https://github.com/kokkos/kokkos/releases/download/5.0.0/kokkos-5.0.0.tar.gz
    URL_HASH SHA256=c45f3e19c3eb71fc8b7210cb04cac658015fc1839e7cc0571f7406588ff9bcef
    SOURCE_DIR "${MINI_MINIPIC_KOKKOS_SOURCE_DIRECTORY}"
  )
  FetchContent_MakeAvailable(kokkos)
endif()

# __________________________________________________________________
# Create executable with options

add_executable(
  minipic
  "src/common/Params.cpp"
  "src/common/Tools.cpp"
  "${SETUP_FILE}"
  "${IMPLEMENTATION_DIRECTORY}/Managers.cpp"
  "${IMPLEMENTATION_DIRECTORY}/Operators.cpp"
  "src/main.cpp"
)
target_include_directories(
  minipic
  PUBLIC
    "src/common"
)
target_compile_definitions(
  minipic
  PUBLIC
    $<$<BOOL:${MINI_MINIPIC_DEBUG}>:MINI_MINIPIC_DEBUG>
    $<$<BOOL:${MINI_MINIPIC_KOKKOS_SCATTER_VIEW}>:MINI_MINIPIC_KOKKOS_SCATTER_VIEW>
)
target_link_libraries(
  minipic
  PRIVATE
    Kokkos::kokkos
)

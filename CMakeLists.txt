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
 message( FATAL_ERROR "In-source builds are not allowed. You should create a separate directory for build files and delete CMakeCache.txt." )
endif()

# __________________________________________________________________
# Options

# Debug, Release, RelWithDebInfo and MinSizeRel
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
option(MINIPIC_DEBUG "Debug messages" OFF)
option(MINIPIC_WARNING "Warning mode" OFF)
option(MINIPIC_UNIFIED_MEMORY "Use unified memory" OFF)
set(MINIPIC_IMPLEMENTATION "exercise" CACHE STRING "Which implementation to use (must be an existing directory in src/)")
set(MINIPIC_SETUP "antenna" CACHE STRING "Which setup to build and run with (must be the stem name of a valid .cpp file in src/setups/)")

# _________________________________________________________________
# Debug mode

message("Debug mode: ${MINIPIC_DEBUG}")

# _________________________________________________________________
# Warning mode

if(MINIPIC_WARNING)
 set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
endif()
message("Warning mode: ${MINIPIC_WARNING}")

# _________________________________________________________________
# Unified memory

message("Unified memory: ${MINIPIC_UNIFIED_MEMORY}")

# _________________________________________________________________
# Implementation

set(IMPLEMENTATION_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/${MINIPIC_IMPLEMENTATION}")
if(NOT EXISTS "${IMPLEMENTATION_DIRECTORY}")
  message(FATAL_ERROR "Cannot find implementation '${MINIPIC_IMPLEMENTATION}' in src/")
endif()
message("Implementation: ${MINIPIC_IMPLEMENTATION}")

# _________________________________________________________________
# Setup

set(SETUP_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/setups/${MINIPIC_SETUP}.cpp")
if(NOT EXISTS "${SETUP_FILE}")
    message(FATAL_ERROR "Cannot find setup file ${MINIPIC_SETUP}.cpp in src/setups/")
endif()
message("Setup: ${MINIPIC_SETUP}")

message("------------------------------------------------------------------")

# __________________________________________________________________
# Project

project(
  minipic
  LANGUAGES CXX
)

# __________________________________________________________________
# Dependencies

find_package(Kokkos QUIET)
if(Kokkos_FOUND)
  message(STATUS "Kokkos provided as installed: ${Kokkos_DIR} (version \"${Kokkos_VERSION}\")")
  message(STATUS "  With backend: ${Kokkos_DEVICES}")
  message(STATUS "  With options: ${Kokkos_OPTIONS}")
  message(STATUS "  For architectures: ${Kokkos_ARCH}")
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/kokkos/CMakeLists.txt")
  message(STATUS "Kokkos submodule found. Using the submodule.")
  add_subdirectory(external/kokkos)
else()
  message(FATAL_ERROR "Kokkos not found. Try running: git submodule update --init --recursive")
endif()

# __________________________________________________________________
# Create executable with options

add_executable(
  minipic
  "src/common/Params.cpp"
  "src/common/Tools.cpp"
  "${SETUP_FILE}"
  "src/main.cpp"
)
target_include_directories(
  minipic
  PUBLIC
    "src/setups"
    "src/common"
    "${IMPLEMENTATION_DIRECTORY}"
)
target_compile_definitions(
  minipic
  PUBLIC
    $<$<BOOL:${MINIPIC_DEBUG}>:__MINIPIC_DEBUG__>
    $<$<BOOL:${MINIPIC_UNIFIED_MEMORY}>:__MINIPIC_KOKKOS_UNIFIED__>
)
target_link_libraries(
  minipic
  PRIVATE
    Kokkos::kokkos
)

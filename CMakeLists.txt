# __________________________________________________________________
#
# CMakeList for Mini-PIC
# 
# __________________________________________________________________

cmake_minimum_required (VERSION 3.16)

message(" __________________________________________________________________ \n")
message(" CMakeList for Mini-PIC")
message(" __________________________________________________________________ \n")

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
message(" Build type: ${CMAKE_BUILD_TYPE}")

# Debug and test modes
option(MINIPIC_DEBUG "Debug messages" OFF)
option(MINIPIC_WARNING "Warning mode" OFF)
option(MINIPIC_TEST "Simulation test for CI" OFF)
option(MINIPIC_BUILD "Compile minipic" ON)
option(MINIPIC_UNIFIED_MEMORY "Use unified memory" OFF)
set(MINIPIC_IMPLEMENTATION "exercise" CACHE STRING "Which implementation to use")

message(" Minipic compilation: ${MINIPIC_BUILD}")

# _________________________________________________________________
# Debug mode

message(" Debug mode: ${MINIPIC_DEBUG}")

# _________________________________________________________________
# Warning mode

if(MINIPIC_WARNING)
 set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
endif()
message(" Warning mode: ${MINIPIC_WARNING}")

# _________________________________________________________________
# Unified memory

message(" Unified memory: ${MINIPIC_UNIFIED_MEMORY}")

# _________________________________________________________________
# Implementation

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/${MINIPIC_IMPLEMENTATION}")
  message(FATAL_ERROR "Cannot find implementation '${MINIPIC_IMPLEMENTATION}' in src/")
endif()
message(" Implementation: ${MINIPIC_IMPLEMENTATION}")

message(" __________________________________________________________________ \n")

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

# Specific behavior for test mode
if(MINIPIC_TEST)
  set(MAIN_PATH "./main.cpp")
else()
  set(MAIN_PATH "./src/main.cpp")
endif()

# __________________________________________________________________
# Create executable with options

file(GLOB SOURCES_COMMON "${CMAKE_CURRENT_SOURCE_DIR}/src/common/*.cpp")

if(MINIPIC_BUILD)
  add_executable(
    minipic
    ${SOURCES_COMMON}
    ${MAIN_PATH}
  )
  target_include_directories(
    minipic
    PUBLIC
      "src/setups"
      "src/common"
      "src/${MINIPIC_IMPLEMENTATION}"
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
endif()

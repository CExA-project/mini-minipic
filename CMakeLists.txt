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
if( "${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}" )
 message( FATAL_ERROR "FATAL: In-source builds are not allowed. You should create a separate directory for build files and delete CMakeCache.txt." )
endif()

# __________________________________________________________________
# Options

# Debug, Release, RelWithDebInfo and MinSizeRel
if (NOT CMAKE_BUILD_TYPE)
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
option(DEBUG "Debug messages" OFF)
option(WARNING "Warning mode (ON/OFF)" OFF)
option(TEST "Simulation test for CI" OFF)
option(MINIPIC "Compile minipic (ON/OFF)" ON)

message(" Minipic compilation: ${MINIPIC}")

set(EXE_NAME "minipic")
message(" Executable name: ${EXE_NAME}")

set(COMPILE_OPTIONS "")
set(LINK_OPTIONS "")

# _________________________________________________________________
# Debug mode

if (DEBUG)
 message(" Debug mode: ON")
 set(COMPILE_OPTIONS ${COMPILE_OPTIONS} -D__MINIPIC_DEBUG__)
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

message(" Languages: ${LANGUAGES}")

message(" __________________________________________________________________ \n")

project (minipic LANGUAGES ${LANGUAGES})

# __________________________________________________________________
# Sources and compile options

file(GLOB_RECURSE SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/kokkos/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/common/*.cpp")
set(COMPILE_OPTIONS ${COMPILE_OPTIONS} -D__MINIPIC_KOKKOS__ -D__MINIPIC_KOKKOS_COMMON__ -D__MINIPIC_KOKKOS_NON_UNIFIED__)
include_directories(src/setups src/common src/kokkos)

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
    message(WARNING "Kokkos not found. Try running: git submodule update --init --recursive")
endif()

# __________________________________________________________________

message("${CMAKE_CURRENT_SOURCE_DIR}")
message("${CMAKE_SOURCE_DIR}")

# Specific behavior for test mode
if (TEST)
  set(MAIN_PATH "./main.cpp")
else()
 set(MAIN_PATH "./src/main.cpp")
endif()

# __________________________________________________________________
# Create executable with options

if (MINIPIC)

  add_executable(${EXE_NAME} ${SOURCES} ${MAIN_PATH})
  target_compile_options(${EXE_NAME} PRIVATE ${COMPILE_OPTIONS})
  target_link_options(${EXE_NAME} PRIVATE ${LINK_OPTIONS})
  target_link_libraries(${EXE_NAME} PRIVATE Kokkos::kokkos)

endif()

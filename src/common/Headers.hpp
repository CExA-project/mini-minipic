/* _____________________________________________________________________ */
//! \file Backend.hpp

//! \brief determine the best backend to use

/* _____________________________________________________________________ */

#ifndef HEADERS_H
#define HEADERS_H

// #include "Params.hpp"

// _____________________________________________________________________
//
// Backends
// _____________________________________________________________________

// ____________________________________________________________
// OMP and OMP task

#if defined(__MINIPIC_OMP__)

#include "omp.h"
#include <atomic>
#include <deque>
#include <memory>
#include <vector>

#define INLINE inline __attribute__((always_inline))
#define DEVICE_INLINE inline __attribute__((always_inline))

// ____________________________________________________________
// Kokkos

#elif defined(__MINIPIC_KOKKOS_COMMON__)

#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <Kokkos_ScatterView.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#define INLINE inline __attribute__((always_inline))
#define DEVICE_INLINE KOKKOS_INLINE_FUNCTION

#else

#include <memory>
#include <vector>
#define INLINE inline __attribute__((always_inline))
#define DEVICE_INLINE inline __attribute__((always_inline))

#endif

// _____________________________________________________________________
// Types

// using mini_float = double;
#define mini_float double

using namespace std;

// _____________________________________________________________________
// Space class

namespace minipic {

class Host {
public:
  static const int value = 1;
};

class Device {
public:
  static const int value = 2;
};

const Host host;
const Device device;

template <typename T> inline void atomicAdd(T *address, T value) {
#if defined(__MINIPIC_OMP__)
#pragma omp atomic update
  *address += value;
#else
  *address += value;
#endif
}

} // namespace minipic

// onHost  on_host;
// onDevice on_device;

#endif

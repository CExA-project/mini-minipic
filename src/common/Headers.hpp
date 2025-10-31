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
// Kokkos

#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <Kokkos_ScatterView.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#define INLINE inline __attribute__((always_inline))
#define DEVICE_INLINE KOKKOS_INLINE_FUNCTION

// _____________________________________________________________________
// Types

// using mini_float = double;
#define mini_float double

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
  *address += value;
}

} // namespace minipic

// onHost  on_host;
// onDevice on_device;

#endif

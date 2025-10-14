
/* _____________________________________________________________________ */
//! \file vector.h

//! \brief Minipic vector class for backend abstraction

/* _____________________________________________________________________ */

// #pragma once
#ifndef VECTOR_H
#define VECTOR_H

#include "Headers.hpp"

// ______________________________________________________________________
//
//! \brief Class Vector for MiniPIC
// ______________________________________________________________________
template <typename T> class Vector {

public:
  // Number of elements
  unsigned int size_;

  // Main data
#if defined(__MINIPIC_KOKKOS_NON_UNIFIED__)
  Kokkos::View<T *> data_;
  typename decltype(data_)::host_mirror_type data_h_;
#elif defined(__MINIPIC_KOKKOS_UNIFIED__)
  Kokkos::View<T *, Kokkos::SharedSpace> data_;
#endif

  // ______________________________________________________________________
  //
  //! \brief constructors
  // ______________________________________________________________________
  Vector() : size_(0) {}
  Vector(unsigned int size) { allocate("", size); }

  // ______________________________________________________________________
  //
  //! \brief constructor with allocation
  //! \param[in] size allocation size
  //! \param[in] v default value
  // ______________________________________________________________________
  Vector(unsigned int size, T v) {
    allocate("", size);
    fill(v, minipic::host);
    fill(v, minipic::device);
  }

  // ______________________________________________________________________
  //
  //! \brief destructor
  // ______________________________________________________________________
  ~Vector() {

  };

  // ______________________________________________________________________
  //
  //! \brief allocate the data_ object
  // ______________________________________________________________________
  void allocate(std::string name, unsigned int size) {
    size_ = size;
#if defined(__MINIPIC_KOKKOS_NON_UNIFIED__)
    data_ = Kokkos::View<T *>(name, size);
    data_h_ = Kokkos::create_mirror_view(data_);
#elif defined(__MINIPIC_KOKKOS_UNIFIED__)
    data_ = Kokkos::View<T *, Kokkos::SharedSpace>(name, size);
#endif
  }

  // ______________________________________________________________________
  //
  //! \brief [] operator
  //! \return Host data accessor (if not device, point to the host data)
  // ______________________________________________________________________
  INLINE T &operator[](const int i) {
#if defined(__MINIPIC_KOKKOS_NON_UNIFIED__)
    return data_h_(i);
#elif defined(__MINIPIC_KOKKOS_UNIFIED__)
    return data_(i);
#endif
  }

  // ______________________________________________________________________
  //
  //! \brief () operator
  //! \return Host data accessor (if not device, point to the host data)
  // ______________________________________________________________________
  INLINE T &operator()(const int i) {
#if defined(__MINIPIC_KOKKOS_NON_UNIFIED__)
    return data_h_(i);
#elif defined(__MINIPIC_KOKKOS_UNIFIED__)
    return data_(i);
#endif
  }

  // ______________________________________________________________________
  //
  //! \brief Explicit Host data accessor
  //! \return host pointer at index i
  // ______________________________________________________________________
  INLINE T &h(const int i) {
#if defined(__MINIPIC_KOKKOS_NON_UNIFIED__)
    return data_h_(i);
#elif defined(__MINIPIC_KOKKOS_UNIFIED__)
    return data_(i);
#endif
  }

  // ______________________________________________________________________
  //
  //! \brief Get the data pointer
  //! \param[in] space where to keep the data when resizing (must be minipic::host or
  //! minipic::device)
  // ______________________________________________________________________
  template <class T_space> T *get_raw_pointer(const T_space) {

    // Check that T_Space of Class Host or Device
    static_assert(std::is_same<T_space, minipic::Host>::value ||
                    std::is_same<T_space, minipic::Device>::value,
                  "Must be minipic::host or minipic::device");

    // Host
    if constexpr (std::is_same<T_space, minipic::Host>::value) {
#if defined(__MINIPIC_KOKKOS_NON_UNIFIED__)
      return data_h_.data();
#elif defined(__MINIPIC_KOKKOS_UNIFIED__)
      return data_.data();
#endif

      // Device
    } else if constexpr (std::is_same<T_space, minipic::Device>::value) {
      return data_.data();

    } else {
      return nullptr;
    }
  }

  // ______________________________________________________________________
  //
  //! \brief return the size
  //! \return size of the vector
  // ______________________________________________________________________
  INLINE T size() { return size_; }

  // ______________________________________________________________________
  //
  //! \brief resize the vector to the new size
  //! \param[in] new_size new vector size
  //! \param[in] space where to keep the data when resizing (must be minipic::host or
  //! minipic::device)
  // ______________________________________________________________________
  template <class T_space> void resize(const unsigned int new_size, const T_space) {

    // Check that T_Space of Class Host or Device
    static_assert(std::is_same<T_space, minipic::Host>::value ||
                    std::is_same<T_space, minipic::Device>::value,
                  "Must be minipic::host or minipic::device");

#if defined(__MINIPIC_KOKKOS_NON_UNIFIED__)
    Kokkos::resize(data_, new_size);
    Kokkos::resize(data_h_, new_size);
#elif defined(__MINIPIC_KOKKOS_UNIFIED__)
    Kokkos::resize(data_, new_size);
    size_ = new_size;
#endif
}
  // ______________________________________________________________________
  //
  //! \brief resize the vector to the new size
  //! \param[in] new_size new vector size
  //! \param[in] value value used to initialize the new elements
  //! \param[in] space where to preserve data (minipic::host or minipic::device)
  //! \tparam T_space class of the space
  // ______________________________________________________________________
  template <class T_space> void resize(const unsigned int new_size, T value, const T_space space) {

    resize(new_size, space); // j'appele la méthode d'avant

#if defined(__MINIPIC_KOKKOS_NON_UNIFIED__)
    for (auto ip = size_; ip < new_size; ++ip) {
      data_h_(ip) = value;
    }
#elif defined(__MINIPIC_KOKKOS_UNIFIED__)
    for (auto ip = size_; ip < new_size; ++ip) {
      data_(ip) = value;
#endif
  }

  // ______________________________________________________________________
  //
  //! \brief clear the content, equivalent to size_ = 0
  //! If the raw object has a clear method, we call it
  // ______________________________________________________________________
  void clear() {
    size_ = 0;
  }

  // ______________________________________________________________________
  //
  //! \brief fill the vector with the given value
  // ______________________________________________________________________
  template <class T_space> void fill(const T v, const T_space) {
    // Check that T_Space of Class Host or Device
    static_assert(std::is_same<T_space, minipic::Host>::value ||
                    std::is_same<T_space, minipic::Device>::value,
                  "Must be minipic::host or minipic::device");

    // Host
    if constexpr (std::is_same<T_space, minipic::Host>::value) {
#if defined(__MINIPIC_KOKKOS_NON_UNIFIED__)
      // Fill on host
      for (auto i = 0; i < size_; ++i) {
        data_h_(i) = v;
      }

#elif defined(__MINIPIC_KOKKOS_UNIFIED__)
      Kokkos::Experimental::fill(Kokkos::DefaultHostExecutionSpace(),
                                 Kokkos::Experimental::begin(data_),
                                 Kokkos::Experimental::end(data_),
                                 v);
      Kokkos::fence();

#endif

    } else if constexpr (std::is_same<T_space, minipic::Device>::value) {

#if defined(__MINIPIC_KOKKOS__)
      // Kokkos::Experimental::fill(Kokkos::DefaultHostExecutionSpace(), data_m.h_view, 0.);

      // Fill on device
      auto& data_ref = data_;
      Kokkos::parallel_for(size_, KOKKOS_LAMBDA(const int ip) { data_ref(ip) = v; });

      Kokkos::fence();

#elif defined(__MINIPIC_KOKKOS_UNIFIED__)

      Kokkos::Experimental::fill(Kokkos::DefaultExecutionSpace(),
                                 Kokkos::Experimental::begin(data_),
                                 Kokkos::Experimental::end(data_),
                                 v);

      Kokkos::fence();
#endif

    } else {
      std::cerr << "Vector::sum: Invalid space" << std::endl;
    }
  }

  // _________________________________________________________
  //
  //! \brief sum of the vector
  //! \param[in] power power of the sum
  //! \param[in] space where to perform the reduction (host or device)
  // _________________________________________________________
  template <class T_space> T sum(const int power, T_space) {
    T sum = 0;

    // ---> Host case
    if constexpr (std::is_same<T_space, minipic::Host>::value) {

#if defined(__MINIPIC_KOKKOS_NON_UNIFIED__)

      typedef Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace> range_policy;

      auto& data_ref = data_h_;
      Kokkos::parallel_reduce(
        "sum",
        range_policy(0, size_),
        KOKKOS_LAMBDA(const int i, T &lsum) { lsum += Kokkos::pow(data_ref(i), power); },
        sum);

      Kokkos::fence();

#elif defined(__MINIPIC_KOKKOS_UNIFIED__)

      typedef Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace> range_policy;

      auto& data_ref = data_;
      Kokkos::parallel_reduce(
        "sum",
        range_policy(0, size_),
        KOKKOS_LAMBDA(const int i, T &lsum) { lsum += Kokkos::pow(data_ref(i), power); },
        sum);

      Kokkos::fence();
#endif

      // ---> Device case
    } else if constexpr (std::is_same<T_space, minipic::Device>::value) {

#if defined(__MINIPIC_KOKKOS__)
      typename Kokkos::View<T *> view = data_;
#elif defined(__MINIPIC_KOKKOS_UNIFIED__)
      typename Kokkos::View<T *, Kokkos::SharedSpace> view = data_;
#endif

      Kokkos::parallel_reduce(
        "sum",
        size_,
        KOKKOS_LAMBDA(const int i, T &lsum) { lsum += Kokkos::pow(view(i), power); },
        sum);

      Kokkos::fence();

    } else {
      std::cerr << "Vector::sum: Invalid space" << std::endl;
    }

    return sum;
  }

  // _________________________________________________________
  //
  //! \brief sync host <-> device
  // _________________________________________________________
  template <class from, class to> void sync(const from, const to) {

    // Check the combination of from and to:
    // - from is minipic::Host then to is minipic::Device
    // - from is minipic::Device then to is minipic::Host
    static_assert(
      (std::is_same<from, minipic::Host>::value && std::is_same<to, minipic::Device>::value) ||
        (std::is_same<from, minipic::Device>::value && std::is_same<to, minipic::Host>::value),
      "Vector::sync: Invalid combination of from and to");

    // Host -> Device
    if constexpr (std::is_same<from, minipic::Host>::value &&
                  std::is_same<to, minipic::Device>::value) {

#if defined(__MINIPIC_KOKKOS_NON_UNIFIED__)
      Kokkos::deep_copy(data_, data_h_);
#endif

      // Device -> Host
    } else if constexpr (std::is_same<from, minipic::Device>::value &&
                         std::is_same<to, minipic::Host>::value) {

#if defined(__MINIPIC_KOKKOS_NON_UNIFIED__)
      Kokkos::deep_copy(data_h_, data_);
#endif
    }
  }
};

// _________________________________________________________________________
// Shortcuts

#if defined(__MINIPIC_KOKKOS__)

using vector_t        = Kokkos::View<mini_float *, Kokkos::DefaultHostExecutionSpace::memory_space>;
using device_vector_t = Kokkos::View<mini_float *>;

#elif defined(__MINIPIC_KOKKOS_UNIFIED__)

using vector_t        = Kokkos::View<mini_float *, Kokkos::SharedSpace>;
using device_vector_t = Kokkos::View<mini_float *, Kokkos::SharedSpace>;

#endif

#endif // VECTOR_H

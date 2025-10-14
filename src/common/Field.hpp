
/* _____________________________________________________________________ */
//! \file Field.hpp

//! \brief class representing a 3D Field array

/* _____________________________________________________________________ */

// #pragma once
#ifndef FIELD_H
#define FIELD_H

#include "Headers.hpp"
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>

// _________________________________________________________________________________________
//! \brief Data structure, store a 3D field

template <typename T> class Field {
public:
  // _________________________________________________________________________________________
  // Variable members

  //! Name of the field
  std::string name_m;

  //! Sizes in each dimension
  int nx_m, ny_m, nz_m;

  //! Factorized sizes
  int nynz_;

  //! Primal 0 / dual 1
  int dual_x_m, dual_y_m, dual_z_m;

  //! Data linearized, 3rd dimension faster

#if defined(__MINIPIC_KOKKOS__)

  Kokkos::View<T ***> data_m;
  typename decltype(data_m)::host_mirror_type data_m_h;

#elif defined(__MINIPIC_KOKKOS_UNIFIED__)

  Kokkos::View<T ***, Kokkos::SharedSpace> data_m;

#endif

  // _________________________________________________________________________________________
  // Methods

  // _________________________________________________________________________________________
  //! \brief Default constructor - create an empty field
  // _________________________________________________________________________________________
  Field() : name_m("empty"), nx_m(0), ny_m(0), nz_m(0), dual_x_m(0), dual_y_m(0), dual_z_m(0) {
    nynz_ = 0;
  }

  // _________________________________________________________________________________________
  //! \brief Constructor - allocate memory for the 3D field with 0 values
  //! \param nx number of grid points in the x direction
  //! \param ny number of grid points in the y direction
  //! \param nz number of grid points in the z direction
  //! \param v default value to fill the field
  //! \param dual_x primal or dual in the x direction
  //! \param dual_y primal or dual in the y direction
  //! \param dual_z primal or dual in the z direction
  //! \param name name of the field
  // _________________________________________________________________________________________
  Field(const int nx,
        const int ny,
        const int nz,
        const T v,
        const int dual_x,
        const int dual_y,
        const int dual_z,
        const std::string name) {
    allocate(nx, ny, nz, v, dual_x, dual_y, dual_z, name);
  }

  // _________________________________________________________________________________________
  //! \brief destructor
  // _________________________________________________________________________________________
  ~Field() {
  }

  // _________________________________________________________________________________________
  //
  //! \brief deep copy constructor
  // _________________________________________________________________________________________
  Field(const Field &f) {
    nx_m     = f.nx_m;
    ny_m     = f.ny_m;
    nz_m     = f.nz_m;
    nynz_    = f.nynz_;
    name_m   = f.name_m;
    dual_x_m = f.dual_x_m;
    dual_y_m = f.dual_y_m;
    dual_z_m = f.dual_z_m;

    data_m = f.data_m;
  }

  // _________________________________________________________________________________________
  //
  //! \brief Get 1d index from 3d indexes
  //! \param i index in the x direction
  //! \param j index in the y direction
  //! \param k index in the z direction
  //! \return the 1d index
  // _________________________________________________________________________________________
  inline __attribute__((always_inline)) int index(const int i, const int j, const int k) const {
    return i * (nz_m * ny_m) + j * (nz_m) + k;
  }

  // _________________________________________________________________________________________
  //
  //! \brief Give the total number of points in the grid
  //! \return the total number of points in the grid
  // _________________________________________________________________________________________
  int size() const { return nx_m * ny_m * nz_m; }

  // _________________________________________________________________________________________
  //
  //! \brief Easiest data accessors using 3D indexes
  //! \param i index in the x direction
  //! \param j index in the y direction
  //! \param k index in the z direction
  //! \return the value of the field at the given indexes
  // _________________________________________________________________________________________
  inline __attribute__((always_inline)) T &
  operator()(const int i, const int j, const int k) noexcept {
#if defined(__MINIPIC_KOKKOS_NON_UNIFIED__)
    return data_m_h(i, j, k);
#elif defined(__MINIPIC_KOKKOS_UNIFIED__)
    return data_m(i, j, k);
#endif
  }

  // _________________________________________________________________________________________
  //
  //! \brief 1d data accessors
  //! \param idx index in the 1d array
  //! \return the value of the field at the given index
  // _________________________________________________________________________________________

  //! \brief return the number of grid points in the x direction
  //! \return return the number of grid points in the x direction
  INLINE int nx() const { return nx_m; }

  //! \brief return the number of grid points in the y direction
  //! \return return the number of grid points in the y direction
  INLINE int ny() const { return ny_m; }

  //! \brief return the number of grid points in the z direction
  //! \return return the number of grid points in the z direction
  INLINE int nz() const { return nz_m; }

  //! \brief return the number of grid points in the y*z direction
  //! \return return the number of grid points in the y*z direction
  INLINE int nynz() const { return nynz_; }

  // _________________________________________________________________________________________
  //
  //! \brief Alloc memory for the 3D field
  //! \param nx number of grid points in the x direction
  //! \param ny number of grid points in the y direction
  //! \param nz number of grid points in the z direction
  //! \param v default value
  //! \param dual_x dual in the x direction
  //! \param dual_y dual in the y direction
  //! \param dual_z dual in the z direction
  //! \param name name of the field
  // _________________________________________________________________________________________
  void allocate(const int nx,
                const int ny,
                const int nz,
                const T v        = 0,
                const int dual_x = 0,
                const int dual_y = 0,
                const int dual_z = 0,
                std::string name = "") {

    nx_m = nx;
    ny_m = ny;
    nz_m = nz;

    nynz_ = ny * nz;

    dual_x_m = dual_x;
    dual_y_m = dual_y;
    dual_z_m = dual_z;
    name_m   = name;

    if (nx_m * ny_m * nz_m == 0) {
      return;
    }

#if defined(__MINIPIC_KOKKOS__)
    data_m = Kokkos::View<T ***>(name, nx, ny, nz);
    data_m_h = create_mirror_view(data_m);
#elif defined(__MINIPIC_KOKKOS_UNIFIED__)
    data_m = Kokkos::View<T ***, Kokkos::SharedSpace>(name, nx, ny, nz);
#endif

    fill(v, minipic::host);
  }

  // _________________________________________________________________________________________
  //
  //! \brief Resize field
  //! \warning This function only preserves the data on the host
  //! \warning Data is not updated on device after resizing
  //! \param nx number of grid points in the x direction
  //! \param ny number of grid points in the y direction
  //! \param nz number of grid points in the z direction
  // _________________________________________________________________________________________
  void resize(const int nx, const int ny, const int nz) {
    nx_m  = nx;
    ny_m  = ny;
    nz_m  = nz;
    nynz_ = ny * nz;
    data_m.resize(nx, ny, nz);
  }

  // _________________________________________________________________________________________
  //
  //! Set the name of the field
  //! \param name name of the field
  // _________________________________________________________________________________________
  void set_name(std::string name) { name_m = name; }

  // _________________________________________________________________________________________
  //
  //! \brief Set all the field at value v
  //! \param v value to set
  //! \param space space where to set the value
  // _________________________________________________________________________________________
  template <class T_space> void fill(const mini_float v, const T_space) {
    // ---> Host case
    if constexpr (std::is_same<T_space, minipic::Host>::value) {
#if defined(__MINIPIC_KOKKOS_NON_UNIFIED__)

      Kokkos::deep_copy(data_m_h, v);

      Kokkos::fence();

#elif defined(__MINIPIC_KOKKOS_UNIFIED__)

      typedef Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Rank<3>>
        mdrange_policy;

      auto& data_ref = data_m;
      Kokkos::parallel_for(
        mdrange_policy({0, 0, 0}, {nx_m, ny_m, nz_m}),
        KOKKOS_LAMBDA(const int ix, const int iy, const int iz) { data_ref(ix, iy, iz) = v; });

      Kokkos::fence();
#endif

      // ---> Device case
    } else if constexpr (std::is_same<T_space, minipic::Device>::value) {
      typedef Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<3>> mdrange_policy;
      auto& data_ref = data_m;
      Kokkos::parallel_for(
        mdrange_policy({0, 0, 0}, {nx_m, ny_m, nz_m}),
        KOKKOS_LAMBDA(const int ix, const int iy, const int iz) { data_ref(ix, iy, iz) = v; });

      Kokkos::fence();
    }
  }

  // _________________________________________________________________________________________
  //
  //! \brief Set all field values to 0
  // _________________________________________________________________________________________
  template <class T_space> void reset(const T_space space) { fill(0, space); }

  // _________________________________________________________________________________________
  //
  //! \brief return the pointer to the data
  //! \param space space where to get the pointer
  //! \tparam T_space execution space
  //! \return return pointer to the first element of the data
  // _________________________________________________________________________________________
  template <class T_space = minipic::Host> T *get_raw_pointer(const T_space) {

    static_assert(std::is_same<T_space, minipic::Host>::value ||
                    std::is_same<T_space, minipic::Device>::value,
                  "T_space must be either minipic::Host or minipic::Device");

#if defined(__MINIPIC_KOKKOS__)
    if constexpr (std::is_same<T_space, minipic::Host>::value) {
      return data_m_h.data();
    } else if constexpr (std::is_same<T_space, minipic::Device>::value) {
      return data_m.data();
    } else {
      return data_m_h.data();
    }
#elif defined(__MINIPIC_KOKKOS_UNIFIED__)
    return data_m.data();
#endif
  }

  // ____________________________________________________________
  //
  //! \brief output the sum of data with power power
  // ____________________________________________________________
  template <class T_space> T sum(const int power, T_space) const {
    T sum = 0;

    // ---> Host case
    if constexpr (std::is_same<T_space, minipic::Host>::value) {

#if defined(__MINIPIC_KOKKOS__)

      auto data_ref = data_m_h;
      typedef Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Rank<3>>
        mdrange_policy;
      Kokkos::parallel_reduce(
        "sum_field_on_host",
        mdrange_policy({0, 0, 0}, {nx_m, ny_m, nz_m}),
        KOKKOS_LAMBDA(const int ix, const int iy, const int iz, T &local_sum) {
          local_sum += Kokkos::pow(data_ref(ix, iy, iz), power);
        },
        sum);

#elif defined(__MINIPIC_KOKKOS_UNIFIED__)

      typedef Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Rank<3>>
        mdrange_policy;

      auto& data_ref = data_m;
      Kokkos::parallel_reduce(
        "sum_field_on_host",
        mdrange_policy({0, 0, 0}, {nx_m, ny_m, nz_m}),
        KOKKOS_LAMBDA(const int ix, const int iy, const int iz, T &local_sum) {
          local_sum += Kokkos::pow(data_ref(ix, iy, iz), power);
        },
        sum);
#endif

      // ---> Device case
    } else if constexpr (std::is_same<T_space, minipic::Device>::value) {
      auto& data_ref = data_m;
      typedef Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<3>> mdrange_policy;
      Kokkos::parallel_reduce(
        "sum_field_on_device",
        mdrange_policy({0, 0, 0}, {nx_m, ny_m, nz_m}),
        KOKKOS_LAMBDA(const int ix, const int iy, const int iz, T &local_sum) {
          local_sum += Kokkos::pow(data_ref(ix, iy, iz), power);
        },
        sum);

      Kokkos::fence();
    }

    return sum;
  }

  // _________________________________________________________________________________________
  //! \brief output the field as a string
  //! \return std::string
  // _________________________________________________________________________________________
  std::string to_string() {
    std::string buffer = "Field " + name_m + "\n";
    buffer += "__________________________________ \n";
    for (auto ix = 0; ix < nx_m; ++ix) {
      buffer += "\n";
      for (auto iy = 0; iy < ny_m; ++iy) {
        buffer += "\n";
        for (auto iz = 0; iz < nz_m; ++iz) {
          // const T field = h(ix, iy, iz);
          // to string with scientific notation
          std::ostringstream out;
          out << std::scientific << this->operator()(ix, iy, iz);
          std::string s = out.str();
          buffer += s + " ";

          // buffer += std::to_string(static_cast<T>(this->operator()(ix, iy, iz))) + " ";
        }
      }
    }
    buffer += "\n __________________________________ \n";
    return buffer;
  }

  // _________________________________________________________________________________________
  //
  //! \brief print all values of the field on host
  // _________________________________________________________________________________________
  void print() {
    std::string buffer = to_string();
    std::cout << buffer << std::endl;
  }

  // _________________________________________________________________________________________
  //
  //! \brief print the sum of the field on host
  // _________________________________________________________________________________________
  void check_sum() {
    T sum = sum();
    std::cout << name_m << " sum: " << sum << std::endl;
  }

  // _________________________________________________________________________________________
  //
  //! \brief Sync Host <-> Device
  // _________________________________________________________________________________________
  template <class T_from, class T_to> void sync(const T_from, const T_to) {
    // ---> Host to Device
    if constexpr (std::is_same<T_from, minipic::Host>::value) {
#if defined(__MINIPIC_KOKKOS__)
      Kokkos::deep_copy(data_m, data_m_h);
#endif
      // ---> Device to Host
    } else if constexpr (std::is_same<T_from, minipic::Device>::value) {
#if defined(__MINIPIC_KOKKOS__)
      Kokkos::deep_copy(data_m_h, data_m);
#endif
    }
  }
};

// _________________________________________________________________________________________
// Shortucts for the different backends

#if defined(__MINIPIC_KOKKOS__)

using device_field_t = Kokkos::View<mini_float ***>;
using field_t        = typename device_field_t::host_mirror_type;

#elif defined(__MINIPIC_KOKKOS_UNIFIED__)

using device_field_t = Kokkos::View<mini_float ***, Kokkos::SharedSpace>;
using field_t        = Kokkos::View<mini_float ***, Kokkos::SharedSpace>;

#endif

#endif // end FIELD_H

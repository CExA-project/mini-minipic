
/* _____________________________________________________________________ */
//! \file Particles_SoA.hpp

//! \brief Particles class to store the particles data as contiguous arrays
//!        for each parameter

//! The parameter `n_particles_m` must be used to store the number of particles,
//! the `std::size` method should not be used for this purpose

/* _____________________________________________________________________ */

#pragma once

#include <cstdio>

#include <iomanip>
#include <math.h>
#include <random>

#include "Vector.hpp"

// ________________________________________________________
//
//! \brief Represent an array of particles for 1 species
// ________________________________________________________
template <typename T> class Particles {
public:
  Particles() : n_particles_m(0) {}
  ~Particles() {}

  //! Inverse of the cell volume, usefull to compute the density
  T inv_cell_volume_m;

  //! Number of particles at init
  int n_particles_m;

  //! Species electric charge
  T charge_m;
  //! Species mass
  T mass_m;
  //! Species temperature
  T temperature_m;

  //! Particles positions in 3D
  Vector<T> x_;
  Vector<T> y_;
  Vector<T> z_;
  //! Particles momentums in 3D
  Vector<T> mx_;
  Vector<T> my_;
  Vector<T> mz_;

  //! Weights | Charge density (scalar)
  //! w0,w1,w2,...
  Vector<T> weight_;

  //! Electric field interpolate
  Vector<T> Ex_;
  Vector<T> Ey_;
  Vector<T> Ez_;
  //! Magnetic field interpolate
  Vector<T> Bx_;
  Vector<T> By_;
  Vector<T> Bz_;

  //! Inverse of Lorentz factor: to avoid recompute gamma between PIC steps
  // Vector<T> gamma_inv_;

  //! This flag when false prevents the allocation of E and B fields
  bool with_electromagnetic_fields_ = true;


  //! \brief Gamma accessor using the momentum
  //! \param[in] ip particle index
  INLINE T gamma(unsigned int ip) {
    return sqrt(1 + mx_.h(ip) * mx_.h(ip) + my_.h(ip) * my_.h(ip) + mz_.h(ip) * mz_.h(ip));
  }

  // __________________________________________________________________________
  //
  //! \brief Alloc memory for a new species
  // __________________________________________________________________________
  void allocate(T q, T m, T t, int n_particles, T icv) {
    inv_cell_volume_m = icv;

    n_particles_m = n_particles;

    // Species properties
    charge_m      = q;
    mass_m        = m;
    temperature_m = t;

    x_.allocate("x", n_particles);
    y_.allocate("y", n_particles);
    z_.allocate("z", n_particles);

    mx_.allocate("mx", n_particles);
    my_.allocate("my", n_particles);
    mz_.allocate("mz", n_particles);

    weight_.allocate("w", n_particles);

    // gamma_inv_.allocate("gamma_inv", n_particles);

    Ex_.allocate("Ex", n_particles);
    Ey_.allocate("Ey", n_particles);
    Ez_.allocate("Ez", n_particles);

    Bx_.allocate("Bx", n_particles);
    By_.allocate("By", n_particles);
    Bz_.allocate("Bz", n_particles);
  }

  // __________________________________________________________________________
  //
  //! \brief Give the number of particles
  // __________________________________________________________________________
  unsigned int size() const { return n_particles_m; }

  // __________________________________________________________________________
  //
  //! \brief Delete all particles properties, keep species properties
  // __________________________________________________________________________
  void clear() {

    x_.clear();
    y_.clear();
    z_.clear();

    mx_.clear();
    my_.clear();
    mz_.clear();

    weight_.clear();

    if (with_electromagnetic_fields_) {
      Ex_.clear();
      Ey_.clear();
      Ez_.clear();

      Bx_.clear();
      By_.clear();
      Bz_.clear();
    }

    n_particles_m = 0;
  }

  // __________________________________________________________________________
  //
  //! \brief Realloc memory to store particles
  // __________________________________________________________________________
  template <class T_space> void resize(int n_particles, const T_space space) {

    // We resize the vectors only if we can gain substantial memory
    // or need more space
    // A particle costs 112 octets

    // This corresponds to a gain of `min_threshold * 112` octets
    const int min_threshold = 500000;

    if (n_particles > n_particles_m || (n_particles_m - n_particles) > min_threshold) {

      x_.resize(n_particles, 0., space);
      y_.resize(n_particles, 0., space);
      z_.resize(n_particles, 0., space);

      mx_.resize(n_particles, 0., space);
      my_.resize(n_particles, 0., space);
      mz_.resize(n_particles, 0., space);

      weight_.resize(n_particles, 0., space);

      if (with_electromagnetic_fields_) {
        Ex_.resize(n_particles, 0., space);
        Ey_.resize(n_particles, 0., space);
        Ez_.resize(n_particles, 0., space);

        Bx_.resize(n_particles, 0., space);
        By_.resize(n_particles, 0., space);
        Bz_.resize(n_particles, 0., space);
      }

      // if (with_gamma_) {
      //   gamma_inv_.resize(n_particles, 0., space);
      // }
    }

    n_particles_m = n_particles;
  }

  // __________________________________________________________________________
  //
  //! \brief Copy particle at index ip in object `particles` at index i of this
  //! \param[in] i index where to put the particles
  //! \param[in] w weight of the particle to add
  //! \param[in] x position of the particle to add
  //! \param[in] y position of the particle to add
  //! \param[in] z position of the particle to add
  //! \param[in] mx momentum of the particle to add
  //! \param[in] my momentum of the particle to add
  //! \param[in] mz momentum of the particle to add
  // __________________________________________________________________________
  void set(int i, T w, T x, T y, T z, T mx, T my, T mz) {

    weight_[i] = w;

    x_[i] = x;
    y_[i] = y;
    z_[i] = z;

    mx_[i] = mx;
    my_[i] = my;
    mz_[i] = mz;

    // gamma_inv_[i] = 1 / sqrt(1 + mx * mx + my * my + mz * mz);

    if (with_electromagnetic_fields_) {
      Ex_[i] = 0;
      Ey_[i] = 0;
      Ez_[i] = 0;

      Bx_[i] = 0;
      By_[i] = 0;
      Bz_[i] = 0;
    }
  }

  // __________________________________________________________________________
  //
  //! \brief Return the total kinetic energy for this particle species
  // __________________________________________________________________________
  template <class T_space> T get_kinetic_energy(T_space) {

    T kinetic_energy = 0;

    if constexpr (std::is_same<T_space, minipic::Device>::value) {

      device_vector_t w  = weight_.data_;
      device_vector_t mx = mx_.data_;
      device_vector_t my = my_.data_;
      device_vector_t mz = mz_.data_;

      Kokkos::parallel_reduce(
        "kinetic_energy_on_device",
        n_particles_m,
        KOKKOS_LAMBDA(const int ip, T &lsum) {
          const T gamma = sqrt(1. + mx(ip) * mx(ip) + my(ip) * my(ip) + mz(ip) * mz(ip));
          lsum += w(ip) * (gamma - 1.);
        },
        kinetic_energy);

      Kokkos::fence();

    } else {

      kinetic_energy = get_kinetic_energy_on_host();
    }

    return kinetic_energy * mass_m;
  }

  // __________________________________________________________________________
  //
  //! \brief data transfer host <-> device
  // __________________________________________________________________________
  template <class T_from, class T_to> void sync(const T_from from, const T_to to) {

    weight_.sync(from, to);

    x_.sync(from, to);
    y_.sync(from, to);
    z_.sync(from, to);

    mx_.sync(from, to);
    my_.sync(from, to);
    mz_.sync(from, to);

    // gamma_inv_.sync(from, to);

    Ex_.sync(from, to);
    Ey_.sync(from, to);
    Ez_.sync(from, to);

    Bx_.sync(from, to);
    By_.sync(from, to);
    Bz_.sync(from, to);
  }

  // __________________________________________________________________________
  //
  //! \brief Print all particles properties
  // __________________________________________________________________________
  void print() {
    for (int ip = 0; ip < n_particles_m; ++ip) {
      std::cerr << "" << ip << " - " << x_.h(ip) << " " << y_.h(ip) << " " << z_.h(ip)
                << " mx: " << mx_.h(ip) << " my: " << my_.h(ip) << " mz: " << mz_.h(ip) << std::endl;
    }
  }

  // __________________________________________________________________________
  //
  //! \brief Check all particles properties
  // __________________________________________________________________________
  void check(T xmin, T xmax, T ymin, T ymax, T zmin, T zmax) {

    for (int ip = 0; ip < n_particles_m; ++ip) {

      if ((x_.h(ip) <= xmin) || (x_.h(ip) >= xmax) || (y_.h(ip) <= ymin) || (y_.h(ip) >= ymax) ||
          (z_.h(ip) <= zmin) || (z_.h(ip) >= zmax)) {
        std::cerr << "Particle: " << ip << "/" << n_particles_m << std::endl;
        std::cerr << " x: " << x_.h(ip) << " [" << xmin << " " << xmax << "]" << std::endl;
        std::cerr << " y: " << y_.h(ip) << " [" << ymin << " " << ymax << "]" << std::endl;
        std::cerr << " z: " << z_.h(ip) << " [" << zmin << " " << zmax << "]" << std::endl;
        std::cerr << " mx: " << mx_.h(ip) << " my: " << my_.h(ip) << " mz: " << mz_.h(ip) << std::endl;
      }
    }
  }

  // __________________________________________________________________________
  //
  //! \brief Print all sums
  // __________________________________________________________________________
  void check_sum() {

    T x_sum = 0;
    T y_sum = 0;
    T z_sum = 0;

    T mx_sum = 0;
    T my_sum = 0;
    T mz_sum = 0;

    // T gamma_inv_sum = 0;

    T Ex_sum = 0;
    T Ey_sum = 0;
    T Ez_sum = 0;

    T Bx_sum = 0;
    T By_sum = 0;
    T Bz_sum = 0;

    for (int ip = 0; ip < n_particles_m; ++ip) {

      x_sum += std::abs(x_.h(ip));
      y_sum += std::abs(y_.h(ip));
      z_sum += std::abs(z_.h(ip));

      mx_sum += std::abs(mx_.h(ip));
      my_sum += std::abs(my_.h(ip));
      mz_sum += std::abs(mz_.h(ip));

      // gamma_inv_sum += std::abs(gamma_inv_h(ip));

      Ex_sum += std::abs(Ex_.h(ip));
      Ey_sum += std::abs(Ey_.h(ip));
      Ez_sum += std::abs(Ez_.h(ip));

      Bx_sum += std::abs(Bx_.h(ip));
      By_sum += std::abs(By_.h(ip));
      Bz_sum += std::abs(Bz_.h(ip));
    }

    std::cerr << std::scientific << std::setprecision(15) << "x sum: " << x_sum
              << " - y sum: " << x_sum << " - z sum: " << x_sum << " - mx sum: " << mx_sum
              << " - my sum: " << my_sum << " - mz sum: "
              << mz_sum
              // << " - gamma inv sum: " << gamma_inv_sum
              << " - Ex: " << Ex_sum << " - Ey: " << Ey_sum << " - Ez: " << Ez_sum
              << " - Bx: " << Bx_sum << " - By: " << By_sum << " - Bz: " << Bz_sum << std::endl;
  }

private:
  // __________________________________________________________________________
  //
  //! \brief Return the total kinetic energy for this particle species
  // __________________________________________________________________________
  T get_kinetic_energy_on_host() {
    T kinetic_energy = 0;

    for (size_t ip = 0; ip < size(); ++ip) {
      const T gamma = sqrt(1. + mx_.h(ip) * mx_.h(ip) + my_.h(ip) * my_.h(ip) + mz_.h(ip) * mz_.h(ip));
      kinetic_energy += weight_.h(ip) * (gamma - 1.);
    }

    return kinetic_energy;
  }
};


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

  //! Type of the Kokkos Views used to store particles information on the device
  using view_t = Kokkos::View<T*>;

  //! Type of the Kokkos Views used to store particles information on the host
  using hostview_t = typename view_t::host_mirror_type;

  //! Particles positions in 3D
  view_t x_;
  hostview_t x_h_;
  view_t y_;
  hostview_t y_h_;
  view_t z_;
  hostview_t z_h_;
  //! Particles momentums in 3D
  view_t mx_;
  hostview_t mx_h_;
  view_t my_;
  hostview_t my_h_;
  view_t mz_;
  hostview_t mz_h_;

  //! Weights | Charge density (scalar)
  //! w0,w1,w2,...
  view_t weight_;
  hostview_t weight_h_;

  //! Electric field interpolate
  view_t Ex_;
  hostview_t Ex_h_;
  view_t Ey_;
  hostview_t Ey_h_;
  view_t Ez_;
  hostview_t Ez_h_;
  //! Magnetic field interpolate
  view_t Bx_;
  hostview_t Bx_h_;
  view_t By_;
  hostview_t By_h_;
  view_t Bz_;
  hostview_t Bz_h_;

  //! This flag when false prevents the allocation of E and B fields
  bool with_electromagnetic_fields_ = true;

  //! \brief Gamma accessor using the momentum
  //! \param[in] ip particle index
  INLINE T gamma(unsigned int ip) {
    return sqrt(1 + mx_h_(ip) * mx_h_(ip) + my_h_(ip) * my_h_(ip) + mz_h_(ip) * mz_h_(ip));
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

    x_ = Kokkos::View<T *>("x", n_particles);
    x_h_ = Kokkos::create_mirror_view(x_);
    y_ = Kokkos::View<T *>("y", n_particles);
    y_h_ = Kokkos::create_mirror_view(y_);
    z_ = Kokkos::View<T *>("z", n_particles);
    z_h_ = Kokkos::create_mirror_view(z_);

    mx_ = Kokkos::View<T *>("mx", n_particles);
    mx_h_ = Kokkos::create_mirror_view(mx_);
    my_ = Kokkos::View<T *>("my", n_particles);
    my_h_ = Kokkos::create_mirror_view(my_);
    mz_ = Kokkos::View<T *>("mz", n_particles);
    mz_h_ = Kokkos::create_mirror_view(mz_);

    weight_ = Kokkos::View<T *>("weight", n_particles);
    weight_h_ = Kokkos::create_mirror_view(weight_);

    // gamma_inv_.allocate("gamma_inv", n_particles);

    Ex_ = Kokkos::View<T *>("Ex", n_particles);
    Ex_h_ = Kokkos::create_mirror_view(Ex_);
    Ey_ = Kokkos::View<T *>("Ey", n_particles);
    Ey_h_ = Kokkos::create_mirror_view(Ey_);
    Ez_ = Kokkos::View<T *>("Ez", n_particles);
    Ez_h_ = Kokkos::create_mirror_view(Ez_);

    Bx_ = Kokkos::View<T *>("Bx", n_particles);
    Bx_h_ = Kokkos::create_mirror_view(Bx_);
    By_ = Kokkos::View<T *>("By", n_particles);
    By_h_ = Kokkos::create_mirror_view(By_);
    Bz_ = Kokkos::View<T *>("Bz", n_particles);
    Bz_h_ = Kokkos::create_mirror_view(Bz_);
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
    Kokkos::resize(x_, 0);
    Kokkos::resize(x_h_, 0);
    Kokkos::resize(y_, 0);
    Kokkos::resize(y_h_, 0);
    Kokkos::resize(z_, 0);
    Kokkos::resize(z_h_, 0);

    Kokkos::resize(mx_, 0);
    Kokkos::resize(mx_h_, 0);
    Kokkos::resize(my_, 0);
    Kokkos::resize(my_h_, 0);
    Kokkos::resize(mz_, 0);
    Kokkos::resize(mz_h_, 0);

    Kokkos::resize(weight_, 0);
    Kokkos::resize(weight_h_, 0);

    if (with_electromagnetic_fields_) {
      Kokkos::resize(Ex_, 0);
      Kokkos::resize(Ex_h_, 0);
      Kokkos::resize(Ey_, 0);
      Kokkos::resize(Ey_h_, 0);
      Kokkos::resize(Ez_, 0);
      Kokkos::resize(Ez_h_, 0);

      Kokkos::resize(Bx_, 0);
      Kokkos::resize(Bx_h_, 0);
      Kokkos::resize(By_, 0);
      Kokkos::resize(By_h_, 0);
      Kokkos::resize(Bz_, 0);
      Kokkos::resize(Bz_h_, 0);
    }

    n_particles_m = 0;
  }

  // __________________________________________________________________________
  //
  //! \brief Realloc memory to store particles
  // __________________________________________________________________________
  void resize(int n_particles) {

    // We resize the vectors only if we can gain substantial memory
    // or need more space
    // A particle costs 112 octets

    // This corresponds to a gain of `min_threshold * 112` octets
    const int min_threshold = 500000;

    if (n_particles > n_particles_m || (n_particles_m - n_particles) > min_threshold) {

      Kokkos::resize(x_, n_particles);
      Kokkos::deep_copy(x_, 0.);
      Kokkos::resize(x_h_, n_particles);
      Kokkos::deep_copy(x_h_, 0.);
      Kokkos::resize(y_, n_particles);
      Kokkos::deep_copy(y_, 0.);
      Kokkos::resize(y_h_, n_particles);
      Kokkos::deep_copy(y_h_, 0.);
      Kokkos::resize(z_, n_particles);
      Kokkos::deep_copy(z_, 0.);
      Kokkos::resize(z_h_, n_particles);
      Kokkos::deep_copy(z_h_, 0.);

      Kokkos::resize(mx_, n_particles);
      Kokkos::deep_copy(mx_, 0.);
      Kokkos::resize(mx_h_, n_particles);
      Kokkos::deep_copy(mx_h_, 0.);
      Kokkos::resize(my_, n_particles);
      Kokkos::deep_copy(my_, 0.);
      Kokkos::resize(my_h_, n_particles);
      Kokkos::deep_copy(my_h_, 0.);
      Kokkos::resize(mz_, n_particles);
      Kokkos::deep_copy(mz_, 0.);
      Kokkos::resize(mz_h_, n_particles);
      Kokkos::deep_copy(mz_h_, 0.);

      if (with_electromagnetic_fields_) {
        Kokkos::resize(Ex_, n_particles);
        Kokkos::deep_copy(Ex_, 0.);
        Kokkos::resize(Ex_h_, n_particles);
        Kokkos::deep_copy(Ex_h_, 0.);
        Kokkos::resize(Ey_, n_particles);
        Kokkos::deep_copy(Ey_, 0.);
        Kokkos::resize(Ey_h_, n_particles);
        Kokkos::deep_copy(Ey_h_, 0.);
        Kokkos::resize(Ez_, n_particles);
        Kokkos::deep_copy(Ez_, 0.);
        Kokkos::resize(Ez_h_, n_particles);
        Kokkos::deep_copy(Ez_h_, 0.);

        Kokkos::resize(Bx_, n_particles);
        Kokkos::deep_copy(Bx_, 0.);
        Kokkos::resize(Bx_h_, n_particles);
        Kokkos::deep_copy(Bx_h_, 0.);
        Kokkos::resize(By_, n_particles);
        Kokkos::deep_copy(By_, 0.);
        Kokkos::resize(By_h_, n_particles);
        Kokkos::deep_copy(By_h_, 0.);
        Kokkos::resize(Bz_, n_particles);
        Kokkos::deep_copy(Bz_, 0.);
        Kokkos::resize(Bz_h_, n_particles);
        Kokkos::deep_copy(Bz_h_, 0.);
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

    x_h_[i] = x;
    y_h_[i] = y;
    z_h_[i] = z;

    mx_h_[i] = mx;
    my_h_[i] = my;
    mz_h_[i] = mz;

    // gamma_inv_[i] = 1 / sqrt(1 + mx * mx + my * my + mz * mz);

    if (with_electromagnetic_fields_) {
      Ex_h_[i] = 0;
      Ey_h_[i] = 0;
      Ez_h_[i] = 0;

      Bx_h_[i] = 0;
      By_h_[i] = 0;
      Bz_h_[i] = 0;
    }

    sync(minipic::host, minipic::device);
  }

  // __________________________________________________________________________
  //
  //! \brief Return the total kinetic energy for this particle species
  // __________________________________________________________________________
  template <class T_space> T get_kinetic_energy(T_space) {

    T kinetic_energy = 0;

    if constexpr (std::is_same<T_space, minipic::Device>::value) {
      auto w  = weight_;
      auto mx = mx_;
      auto my = my_;
      auto mz = mz_;

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
  template <class from, class to> void sync(const from&, const to&) {
    // Check the combination of from and to:
    // - from is minipic::Host then to is minipic::Device
    // - from is minipic::Device then to is minipic::Host
    static_assert(
      !std::is_same<from, to>::value,
      "Particles::sync: Invalid combination of from and to");

    if constexpr (std::is_same<from, minipic::Host>::value &&
                  std::is_same<to, minipic::Device>::value) {
      // Host -> Device
      Kokkos::deep_copy(x_, x_h_);
      Kokkos::deep_copy(y_, y_h_);
      Kokkos::deep_copy(z_, z_h_);

      Kokkos::deep_copy(mx_, mx_h_);
      Kokkos::deep_copy(my_, my_h_);
      Kokkos::deep_copy(mz_, mz_h_);

      Kokkos::deep_copy(weight_, weight_h_);

      Kokkos::deep_copy(Ex_, Ex_h_);
      Kokkos::deep_copy(Ey_, Ey_h_);
      Kokkos::deep_copy(Ez_, Ez_h_);

      Kokkos::deep_copy(Bx_, Bx_h_);
      Kokkos::deep_copy(By_, By_h_);
      Kokkos::deep_copy(Bz_, Bz_h_);
    } else if constexpr (std::is_same<from, minipic::Device>::value &&
                         std::is_same<to, minipic::Host>::value) {
      // Device -> Host
      Kokkos::deep_copy(x_h_, x_);
      Kokkos::deep_copy(y_h_, y_);
      Kokkos::deep_copy(z_h_, z_);

      Kokkos::deep_copy(mx_h_, mx_);
      Kokkos::deep_copy(my_h_, my_);
      Kokkos::deep_copy(mz_h_, mz_);

      Kokkos::deep_copy(weight_h_, weight_);

      Kokkos::deep_copy(Ex_h_, Ex_);
      Kokkos::deep_copy(Ey_h_, Ey_);
      Kokkos::deep_copy(Ez_h_, Ez_);

      Kokkos::deep_copy(Bx_h_, Bx_);
      Kokkos::deep_copy(By_h_, By_);
      Kokkos::deep_copy(Bz_h_, Bz_);

    }
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
      const T gamma = sqrt(1. + mx_h_(ip) * mx_h_(ip) + my_h_(ip) * my_h_(ip) + mz_h_(ip) * mz_h_(ip));
      kinetic_energy += weight_h_(ip) * (gamma - 1.);
    }

    return kinetic_energy;
  }
};

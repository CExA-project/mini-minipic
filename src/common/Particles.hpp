
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
class Particles {
public:
  Particles() : n_particles_m(0) {}
  ~Particles() {}

  //! Inverse of the cell volume, usefull to compute the density
  double inv_cell_volume_m;

  //! Number of particles at init
  int n_particles_m;

  //! Species electric charge
  double charge_m;
  //! Species mass
  double mass_m;
  //! Species temperature
  double temperature_m;

  //! Type of the Kokkos Views used to store particles information on the device
  using view_t = Kokkos::View<double*>;

  //! Type of the Kokkos Views used to store particles information on the host
  using hostview_t = typename view_t::host_mirror_type;

  //! Particles positions in 3D
  view_t x_m;
  hostview_t x_h_m;
  view_t y_m;
  hostview_t y_h_m;
  view_t z_m;
  hostview_t z_h_m;
  //! Particles momentums in 3D
  view_t mx_m;
  hostview_t mx_h_m;
  view_t my_m;
  hostview_t my_h_m;
  view_t mz_m;
  hostview_t mz_h_m;

  //! Weights | Charge density (scalar)
  //! w0,w1,w2,...
  view_t weight_m;
  hostview_t weight_h_m;

  //! Electric field interpolate
  view_t Ex_m;
  hostview_t Ex_h_m;
  view_t Ey_m;
  hostview_t Ey_h_m;
  view_t Ez_m;
  hostview_t Ez_h_m;
  //! Magnetic field interpolate
  view_t Bx_m;
  hostview_t Bx_h_m;
  view_t By_m;
  hostview_t By_h_m;
  view_t Bz_m;
  hostview_t Bz_h_m;

  //! This flag when false prevents the allocation of E and B fields
  bool with_electromagnetic_fields_m = true;

  //! \brief Gamma accessor using the momentum
  //! \param[in] ip particle index
  INLINE double gamma(unsigned int ip) {
    return sqrt(1 + mx_h_m(ip) * mx_h_m(ip) + my_h_m(ip) * my_h_m(ip) + mz_h_m(ip) * mz_h_m(ip));
  }

  // __________________________________________________________________________
  //
  //! \brief Alloc memory for a new species
  // __________________________________________________________________________
  void allocate(double q, double m, double t, int n_particles, double icv) {
    inv_cell_volume_m = icv;

    n_particles_m = n_particles;

    // Species properties
    charge_m      = q;
    mass_m        = m;
    temperature_m = t;

    x_m = view_t("x", n_particles);
    x_h_m = Kokkos::create_mirror_view(x_m);
    y_m = view_t("y", n_particles);
    y_h_m = Kokkos::create_mirror_view(y_m);
    z_m = view_t("z", n_particles);
    z_h_m = Kokkos::create_mirror_view(z_m);

    mx_m = view_t("mx", n_particles);
    mx_h_m = Kokkos::create_mirror_view(mx_m);
    my_m = view_t("my", n_particles);
    my_h_m = Kokkos::create_mirror_view(my_m);
    mz_m = view_t("mz", n_particles);
    mz_h_m = Kokkos::create_mirror_view(mz_m);

    weight_m = view_t("weight", n_particles);
    weight_h_m = Kokkos::create_mirror_view(weight_m);

    // gamma_inv_m.allocate("gamma_inv", n_particles);

    Ex_m = view_t("Ex", n_particles);
    Ex_h_m = Kokkos::create_mirror_view(Ex_m);
    Ey_m = view_t("Ey", n_particles);
    Ey_h_m = Kokkos::create_mirror_view(Ey_m);
    Ez_m = view_t("Ez", n_particles);
    Ez_h_m = Kokkos::create_mirror_view(Ez_m);

    Bx_m = view_t("Bx", n_particles);
    Bx_h_m = Kokkos::create_mirror_view(Bx_m);
    By_m = view_t("By", n_particles);
    By_h_m = Kokkos::create_mirror_view(By_m);
    Bz_m = view_t("Bz", n_particles);
    Bz_h_m = Kokkos::create_mirror_view(Bz_m);
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
    Kokkos::resize(x_m, 0);
    Kokkos::resize(x_h_m, 0);
    Kokkos::resize(y_m, 0);
    Kokkos::resize(y_h_m, 0);
    Kokkos::resize(z_m, 0);
    Kokkos::resize(z_h_m, 0);

    Kokkos::resize(mx_m, 0);
    Kokkos::resize(mx_h_m, 0);
    Kokkos::resize(my_m, 0);
    Kokkos::resize(my_h_m, 0);
    Kokkos::resize(mz_m, 0);
    Kokkos::resize(mz_h_m, 0);

    Kokkos::resize(weight_m, 0);
    Kokkos::resize(weight_h_m, 0);

    if (with_electromagnetic_fields_m) {
      Kokkos::resize(Ex_m, 0);
      Kokkos::resize(Ex_h_m, 0);
      Kokkos::resize(Ey_m, 0);
      Kokkos::resize(Ey_h_m, 0);
      Kokkos::resize(Ez_m, 0);
      Kokkos::resize(Ez_h_m, 0);

      Kokkos::resize(Bx_m, 0);
      Kokkos::resize(Bx_h_m, 0);
      Kokkos::resize(By_m, 0);
      Kokkos::resize(By_h_m, 0);
      Kokkos::resize(Bz_m, 0);
      Kokkos::resize(Bz_h_m, 0);
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

      Kokkos::resize(x_m, n_particles);
      Kokkos::deep_copy(x_m, 0.);
      Kokkos::resize(x_h_m, n_particles);
      Kokkos::deep_copy(x_h_m, 0.);
      Kokkos::resize(y_m, n_particles);
      Kokkos::deep_copy(y_m, 0.);
      Kokkos::resize(y_h_m, n_particles);
      Kokkos::deep_copy(y_h_m, 0.);
      Kokkos::resize(z_m, n_particles);
      Kokkos::deep_copy(z_m, 0.);
      Kokkos::resize(z_h_m, n_particles);
      Kokkos::deep_copy(z_h_m, 0.);

      Kokkos::resize(mx_m, n_particles);
      Kokkos::deep_copy(mx_m, 0.);
      Kokkos::resize(mx_h_m, n_particles);
      Kokkos::deep_copy(mx_h_m, 0.);
      Kokkos::resize(my_m, n_particles);
      Kokkos::deep_copy(my_m, 0.);
      Kokkos::resize(my_h_m, n_particles);
      Kokkos::deep_copy(my_h_m, 0.);
      Kokkos::resize(mz_m, n_particles);
      Kokkos::deep_copy(mz_m, 0.);
      Kokkos::resize(mz_h_m, n_particles);
      Kokkos::deep_copy(mz_h_m, 0.);

      if (with_electromagnetic_fields_m) {
        Kokkos::resize(Ex_m, n_particles);
        Kokkos::deep_copy(Ex_m, 0.);
        Kokkos::resize(Ex_h_m, n_particles);
        Kokkos::deep_copy(Ex_h_m, 0.);
        Kokkos::resize(Ey_m, n_particles);
        Kokkos::deep_copy(Ey_m, 0.);
        Kokkos::resize(Ey_h_m, n_particles);
        Kokkos::deep_copy(Ey_h_m, 0.);
        Kokkos::resize(Ez_m, n_particles);
        Kokkos::deep_copy(Ez_m, 0.);
        Kokkos::resize(Ez_h_m, n_particles);
        Kokkos::deep_copy(Ez_h_m, 0.);

        Kokkos::resize(Bx_m, n_particles);
        Kokkos::deep_copy(Bx_m, 0.);
        Kokkos::resize(Bx_h_m, n_particles);
        Kokkos::deep_copy(Bx_h_m, 0.);
        Kokkos::resize(By_m, n_particles);
        Kokkos::deep_copy(By_m, 0.);
        Kokkos::resize(By_h_m, n_particles);
        Kokkos::deep_copy(By_h_m, 0.);
        Kokkos::resize(Bz_m, n_particles);
        Kokkos::deep_copy(Bz_m, 0.);
        Kokkos::resize(Bz_h_m, n_particles);
        Kokkos::deep_copy(Bz_h_m, 0.);
      }

      // if (with_gamma_m) {
      //   gamma_inv_m.resize(n_particles, 0., space);
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
  void set(int i, double w, double x, double y, double z, double mx, double my, double mz) {
    weight_h_m[i] = w;

    x_h_m[i] = x;
    y_h_m[i] = y;
    z_h_m[i] = z;

    mx_h_m[i] = mx;
    my_h_m[i] = my;
    mz_h_m[i] = mz;

    // gamma_inv_m[i] = 1 / sqrt(1 + mx * mx + my * my + mz * mz);

    if (with_electromagnetic_fields_m) {
      Ex_h_m[i] = 0;
      Ey_h_m[i] = 0;
      Ez_h_m[i] = 0;

      Bx_h_m[i] = 0;
      By_h_m[i] = 0;
      Bz_h_m[i] = 0;
    }

    sync(minipic::host, minipic::device);
  }

  // __________________________________________________________________________
  //
  //! \brief Return the total kinetic energy for this particle species
  // __________________________________________________________________________
  template <class T_space> double get_kinetic_energy(T_space) {

    double kinetic_energy = 0;

    if constexpr (std::is_same<T_space, minipic::Device>::value) {
      auto w  = weight_m;
      auto mx = mx_m;
      auto my = my_m;
      auto mz = mz_m;

      Kokkos::parallel_reduce(
        "kinetic_energy_on_device",
        n_particles_m,
        KOKKOS_LAMBDA(const int ip, double &lsum) {
          const double gamma = sqrt(1. + mx(ip) * mx(ip) + my(ip) * my(ip) + mz(ip) * mz(ip));
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
      Kokkos::deep_copy(x_m, x_h_m);
      Kokkos::deep_copy(y_m, y_h_m);
      Kokkos::deep_copy(z_m, z_h_m);

      Kokkos::deep_copy(mx_m, mx_h_m);
      Kokkos::deep_copy(my_m, my_h_m);
      Kokkos::deep_copy(mz_m, mz_h_m);

      Kokkos::deep_copy(weight_m, weight_h_m);

      Kokkos::deep_copy(Ex_m, Ex_h_m);
      Kokkos::deep_copy(Ey_m, Ey_h_m);
      Kokkos::deep_copy(Ez_m, Ez_h_m);

      Kokkos::deep_copy(Bx_m, Bx_h_m);
      Kokkos::deep_copy(By_m, By_h_m);
      Kokkos::deep_copy(Bz_m, Bz_h_m);
    } else if constexpr (std::is_same<from, minipic::Device>::value &&
                         std::is_same<to, minipic::Host>::value) {
      // Device -> Host
      Kokkos::deep_copy(x_h_m, x_m);
      Kokkos::deep_copy(y_h_m, y_m);
      Kokkos::deep_copy(z_h_m, z_m);

      Kokkos::deep_copy(mx_h_m, mx_m);
      Kokkos::deep_copy(my_h_m, my_m);
      Kokkos::deep_copy(mz_h_m, mz_m);

      Kokkos::deep_copy(weight_h_m, weight_m);

      Kokkos::deep_copy(Ex_h_m, Ex_m);
      Kokkos::deep_copy(Ey_h_m, Ey_m);
      Kokkos::deep_copy(Ez_h_m, Ez_m);

      Kokkos::deep_copy(Bx_h_m, Bx_m);
      Kokkos::deep_copy(By_h_m, By_m);
      Kokkos::deep_copy(Bz_h_m, Bz_m);

    }
  }

  // __________________________________________________________________________
  //
  //! \brief Print all particles properties
  // __________________________________________________________________________
  void print() {
    for (int ip = 0; ip < n_particles_m; ++ip) {
      std::cerr << "" << ip << " - " << x_h_m(ip) << " " << y_h_m(ip) << " " << z_h_m(ip)
                << " mx: " << mx_h_m(ip) << " my: " << my_h_m(ip) << " mz: " << mz_h_m(ip) << std::endl;
    }
  }

  // __________________________________________________________________________
  //
  //! \brief Check all particles properties
  // __________________________________________________________________________
  void check(double xmin, double xmax, double ymin, double ymax, double zmin, double zmax) {

    for (int ip = 0; ip < n_particles_m; ++ip) {

      if ((x_h_m(ip) <= xmin) || (x_h_m(ip) >= xmax) || (y_h_m(ip) <= ymin) || (y_h_m(ip) >= ymax) ||
          (z_h_m(ip) <= zmin) || (z_h_m(ip) >= zmax)) {
        std::cerr << "Particle: " << ip << "/" << n_particles_m << std::endl;
        std::cerr << " x: " << x_h_m(ip) << " [" << xmin << " " << xmax << "]" << std::endl;
        std::cerr << " y: " << y_h_m(ip) << " [" << ymin << " " << ymax << "]" << std::endl;
        std::cerr << " z: " << z_h_m(ip) << " [" << zmin << " " << zmax << "]" << std::endl;
        std::cerr << " mx: " << mx_h_m(ip) << " my: " << my_h_m(ip) << " mz: " << mz_h_m(ip) << std::endl;
      }
    }
  }

  // __________________________________________________________________________
  //
  //! \brief Print all sums
  // __________________________________________________________________________
  void check_sum() {

    double x_sum = 0;
    double y_sum = 0;
    double z_sum = 0;

    double mx_sum = 0;
    double my_sum = 0;
    double mz_sum = 0;

    // double gamma_inv_sum = 0;

    double Ex_sum = 0;
    double Ey_sum = 0;
    double Ez_sum = 0;

    double Bx_sum = 0;
    double By_sum = 0;
    double Bz_sum = 0;

    for (int ip = 0; ip < n_particles_m; ++ip) {

      x_sum += std::abs(x_h_m(ip));
      y_sum += std::abs(y_h_m(ip));
      z_sum += std::abs(z_h_m(ip));

      mx_sum += std::abs(mx_h_m(ip));
      my_sum += std::abs(my_h_m(ip));
      mz_sum += std::abs(mz_h_m(ip));

      // gamma_inv_sum += std::abs(gamma_inv_h(ip));

      Ex_sum += std::abs(Ex_h_m(ip));
      Ey_sum += std::abs(Ey_h_m(ip));
      Ez_sum += std::abs(Ez_h_m(ip));

      Bx_sum += std::abs(Bx_h_m(ip));
      By_sum += std::abs(By_h_m(ip));
      Bz_sum += std::abs(Bz_h_m(ip));
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
  double get_kinetic_energy_on_host() {
    double kinetic_energy = 0;

    for (size_t ip = 0; ip < size(); ++ip) {
      const double gamma = sqrt(1. + mx_h_m(ip) * mx_h_m(ip) + my_h_m(ip) * my_h_m(ip) + mz_h_m(ip) * mz_h_m(ip));
      kinetic_energy += weight_h_m(ip) * (gamma - 1.);
    }

    return kinetic_energy;
  }
};

/* _____________________________________________________________________ */
//! \file Operators.cpp

//! \brief contains generic kernels for the particle pusher

/* _____________________________________________________________________ */

#include <Kokkos_Core.hpp>

#include "Operators.hpp"

namespace operators {

//! \brief Returns the sum of all elements of a View on the host.
//! \param[in] view View on the host to reduce.
//! \returns Sum of all values.
double sum_host(typename Particles::hostview_t view) {
  double res = 0.f;
  for (std::size_t i=0; i < view.extent(0); ++i) {
    res += view(i);
  }
  return res;
}

//! \brief Returns the sum of all elements of a View on the device
//! \param[in] view View on the device to reduce.
//! \returns Sum of all values.
double sum_device(typename Particles::view_t view) {
  double res;
  Kokkos::parallel_reduce(Kokkos::RangePolicy(0, view.extent(0)),
      KOKKOS_LAMBDA(const int i, double &partial_res){
        partial_res += view(i);
      }, res);

  return res;
}

//! \brief Returns the sum of the power of all elements of a View on the host
//! \param[in] view View on the host to reduce.
//! \param[in] power Value of the exponent.
//! \returns Sum of the power of all values.
double sum_power(ElectroMagn::hostview_t v, const int power) {
  double sum = 0;

  using mdrange_policy = Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Rank<3>>;
  Kokkos::parallel_reduce(
      "sum_field_on_host",
      mdrange_policy({0, 0, 0}, {v.extent(0), v.extent(1), v.extent(2)}),
      KOKKOS_LAMBDA(const int ix, const int iy, const int iz, double &local_sum) {
      local_sum += Kokkos::pow(v(ix, iy, iz), power);
      },
      sum);

  return sum;
}

//! \brief Returns the sum of the power of all elements of a View on the device
//! \param[in] view View on the device to reduce.
//! \param[in] power Value of the exponent.
//! \returns Sum of the power of all values.
double sum_power(ElectroMagn::view_t v, const int power) {
  double sum = 0;

  using mdrange_policy = Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<3>>;
  Kokkos::parallel_reduce(
      "sum_field_on_device",
      mdrange_policy({0, 0, 0}, {v.extent(0), v.extent(1), v.extent(2)}),
      KOKKOS_LAMBDA(const int ix, const int iy, const int iz, double &local_sum) {
      local_sum += Kokkos::pow(v(ix, iy, iz), power);
      },
      sum);

  return sum;
}

//! \brief Interpolation operator: interpolate EM fields from global grid for each particle
//! \param[in] em  global electromagnetic fields
//! \param[in] particles  vector of particle species
void interpolate(ElectroMagn &em, std::vector<Particles> &particles) {
  const auto inv_dx_m = em.inv_dx_m;
  const auto inv_dy_m = em.inv_dy_m;
  const auto inv_dz_m = em.inv_dz_m;

  for (std::size_t is = 0; is < particles.size(); is++) {

    const std::size_t n_particles = particles[is].size();

    ElectroMagn::view_t Ex = em.Ex_m;
    ElectroMagn::view_t Ey = em.Ey_m;
    ElectroMagn::view_t Ez = em.Ez_m;

    ElectroMagn::view_t Bx = em.Bx_m;
    ElectroMagn::view_t By = em.By_m;
    ElectroMagn::view_t Bz = em.Bz_m;

    Particles::view_t x = particles[is].x_m;
    Particles::view_t y = particles[is].y_m;
    Particles::view_t z = particles[is].z_m;

    Particles::view_t Exp = particles[is].Ex_m;
    Particles::view_t Eyp = particles[is].Ey_m;
    Particles::view_t Ezp = particles[is].Ez_m;

    Particles::view_t Bxp = particles[is].Bx_m;
    Particles::view_t Byp = particles[is].By_m;
    Particles::view_t Bzp = particles[is].Bz_m;

    Kokkos::parallel_for(
      n_particles,
      KOKKOS_LAMBDA(const int part) {
        //  [=, &Ex](const int part) {

        // // Calculate normalized positions
        const double ixn = x(part) * inv_dx_m;
        const double iyn = y(part) * inv_dy_m;
        const double izn = z(part) * inv_dz_m;

        // // Compute indexes in global primal grid
        const unsigned int ixp = Kokkos::floor(ixn);
        const unsigned int iyp = Kokkos::floor(iyn);
        const unsigned int izp = Kokkos::floor(izn);

        // Compute indexes in global dual grid
        const unsigned int ixd = Kokkos::floor(ixn + 0.5);
        const unsigned int iyd = Kokkos::floor(iyn + 0.5);
        const unsigned int izd = Kokkos::floor(izn + 0.5);

        // Compute interpolation coeff, p = primal, d = dual

        // Ex (d, p, p)
        {
          const double coeffs[3] = {ixn + 0.5, iyn, izn};

          const double v00 =
            Ex(ixd, iyp, izp) * (1 - coeffs[0]) + Ex(ixd + 1, iyp, izp) * coeffs[0];
          const double v01 =
            Ex(ixd, iyp, izp + 1) * (1 - coeffs[0]) + Ex(ixd + 1, iyp, izp + 1) * coeffs[0];
          const double v10 =
            Ex(ixd, iyp + 1, izp) * (1 - coeffs[0]) + Ex(ixd + 1, iyp + 1, izp) * coeffs[0];
          const double v11 =
            Ex(ixd, iyp + 1, izp + 1) * (1 - coeffs[0]) + Ex(ixd + 1, iyp + 1, izp + 1) * coeffs[0];
          const double v0 = v00 * (1 - coeffs[1]) + v10 * coeffs[1];
          const double v1 = v01 * (1 - coeffs[1]) + v11 * coeffs[1];

          Exp(part) = v0 * (1 - coeffs[2]) + v1 * coeffs[2];
        }

        // Ey (p, d, p)
        {
          const double coeffs[3] = {ixn, iyn + 0.5, izn};

          const double v00 =
            Ey(ixp, iyd, izp) * (1 - coeffs[0]) + Ey(ixp + 1, iyd, izp) * coeffs[0];
          const double v01 =
            Ey(ixp, iyd, izp + 1) * (1 - coeffs[0]) + Ey(ixp + 1, iyd, izp + 1) * coeffs[0];
          const double v10 =
            Ey(ixp, iyd + 1, izp) * (1 - coeffs[0]) + Ey(ixp + 1, iyd + 1, izp) * coeffs[0];
          const double v11 =
            Ey(ixp, iyd + 1, izp + 1) * (1 - coeffs[0]) + Ey(ixp + 1, iyd + 1, izp + 1) * coeffs[0];
          const double v0 = v00 * (1 - coeffs[1]) + v10 * coeffs[1];
          const double v1 = v01 * (1 - coeffs[1]) + v11 * coeffs[1];

          Eyp(part) = v0 * (1 - coeffs[2]) + v1 * coeffs[2];
        }

        // Ez (p, p, d)
        {
          const double coeffs[3] = {ixn, iyn, izn + 0.5};

          const double v00 =
            Ez(ixp, iyp, izd) * (1 - coeffs[0]) + Ez(ixp + 1, iyp, izd) * coeffs[0];
          const double v01 =
            Ez(ixp, iyp, izd + 1) * (1 - coeffs[0]) + Ez(ixp + 1, iyp, izd + 1) * coeffs[0];
          const double v10 =
            Ez(ixp, iyp + 1, izd) * (1 - coeffs[0]) + Ez(ixp + 1, iyp + 1, izd) * coeffs[0];
          const double v11 =
            Ez(ixp, iyp + 1, izd + 1) * (1 - coeffs[0]) + Ez(ixp + 1, iyp + 1, izd + 1) * coeffs[0];
          const double v0 = v00 * (1 - coeffs[1]) + v10 * coeffs[1];
          const double v1 = v01 * (1 - coeffs[1]) + v11 * coeffs[1];

          Ezp(part) = v0 * (1 - coeffs[2]) + v1 * coeffs[2];
        }

        // interpolation magnetic field
        // Bx (p, d, d)
        {
          const double coeffs[3] = {ixn, iyn + 0.5, izn + 0.5};

          const double v00 =
            Bx(ixp, iyd, izd) * (1 - coeffs[0]) + Bx(ixp + 1, iyd, izd) * coeffs[0];
          const double v01 =
            Bx(ixp, iyd, izd + 1) * (1 - coeffs[0]) + Bx(ixp + 1, iyd, izd + 1) * coeffs[0];
          const double v10 =
            Bx(ixp, iyd + 1, izd) * (1 - coeffs[0]) + Bx(ixp + 1, iyd + 1, izd) * coeffs[0];
          const double v11 =
            Bx(ixp, iyd + 1, izd + 1) * (1 - coeffs[0]) + Bx(ixp + 1, iyd + 1, izd + 1) * coeffs[0];
          const double v0 = v00 * (1 - coeffs[1]) + v10 * coeffs[1];
          const double v1 = v01 * (1 - coeffs[1]) + v11 * coeffs[1];

          Bxp(part) = v0 * (1 - coeffs[2]) + v1 * coeffs[2];
        }

        // By (d, p, d)
        {
          const double coeffs[3] = {ixn + 0.5, iyn, izn + 0.5};

          const double v00 =
            By(ixd, iyp, izd) * (1 - coeffs[0]) + By(ixd + 1, iyp, izd) * coeffs[0];
          const double v01 =
            By(ixd, iyp, izd + 1) * (1 - coeffs[0]) + By(ixd + 1, iyp, izd + 1) * coeffs[0];
          const double v10 =
            By(ixd, iyp + 1, izd) * (1 - coeffs[0]) + By(ixd + 1, iyp + 1, izd) * coeffs[0];
          const double v11 =
            By(ixd, iyp + 1, izd + 1) * (1 - coeffs[0]) + By(ixd + 1, iyp + 1, izd + 1) * coeffs[0];
          const double v0 = v00 * (1 - coeffs[1]) + v10 * coeffs[1];
          const double v1 = v01 * (1 - coeffs[1]) + v11 * coeffs[1];

          Byp(part) = v0 * (1 - coeffs[2]) + v1 * coeffs[2];
        }

        // Bz (d, d, p)
        {
          const double coeffs[3] = {ixn + 0.5, iyn + 0.5, izn};

          const double v00 =
            Bz(ixd, iyd, izp) * (1 - coeffs[0]) + Bz(ixd + 1, iyd, izp) * coeffs[0];
          const double v01 =
            Bz(ixd, iyd, izp + 1) * (1 - coeffs[0]) + Bz(ixd + 1, iyd, izp + 1) * coeffs[0];
          const double v10 =
            Bz(ixd, iyd + 1, izp) * (1 - coeffs[0]) + Bz(ixd + 1, iyd + 1, izp) * coeffs[0];
          const double v11 =
            Bz(ixd, iyd + 1, izp + 1) * (1 - coeffs[0]) + Bz(ixd + 1, iyd + 1, izp + 1) * coeffs[0];
          const double v0 = v00 * (1 - coeffs[1]) + v10 * coeffs[1];
          const double v1 = v01 * (1 - coeffs[1]) + v11 * coeffs[1];

          Bzp(part) = v0 * (1 - coeffs[2]) + v1 * coeffs[2];
        }

      } // End for each particle

    ); // end KOKKOS PARALLEL

    Kokkos::fence();

  } // Species loop
}

//! \brief Move the particle in the space, compute with EM fields interpolate.
//! \param[in] particles Vector of particle species.
//! \param[in] dt Time step to use for the pusher.
void push(std::vector<Particles> &particles, double dt) {

  // For each species
  for (std::size_t is = 0; is < particles.size(); is++) {

    const std::size_t n_particles = particles[is].size();

    // q' = dt * (q/2m)
    const double qp = particles[is].charge_m * dt * 0.5 / particles[is].mass_m;

    particles[is].sync(minipic::device, minipic::host);
    Particles::view_t x = particles[is].x_m;
    Particles::view_t y = particles[is].y_m;
    Particles::view_t z = particles[is].z_m;

    Particles::view_t mx = particles[is].mx_m;
    Particles::view_t my = particles[is].my_m;
    Particles::view_t mz = particles[is].mz_m;

    Particles::view_t Exp = particles[is].Ex_m;
    Particles::view_t Eyp = particles[is].Ey_m;
    Particles::view_t Ezp = particles[is].Ez_m;

    Particles::view_t Bxp = particles[is].Bx_m;
    Particles::view_t Byp = particles[is].By_m;
    Particles::view_t Bzp = particles[is].Bz_m;

    Kokkos::parallel_for(
      n_particles,
      KOKKOS_LAMBDA(const int ip) {
        // 1/2 E
        double px = qp * Exp(ip);
        double py = qp * Eyp(ip);
        double pz = qp * Ezp(ip);

        const double ux = mx(ip) + px;
        const double uy = my(ip) + py;
        const double uz = mz(ip) + pz;

        // gamma-factor
        double usq       = (ux * ux + uy * uy + uz * uz);
        double gamma     = Kokkos::sqrt(1 + usq);
        double gamma_inv = qp / gamma;

        // B, T = Transform to rotate the particle
        const double tx  = gamma_inv * Bxp(ip);
        const double ty  = gamma_inv * Byp(ip);
        const double tz  = gamma_inv * Bzp(ip);
        const double tsq = 1. + (tx * tx + ty * ty + tz * tz);
        double tsq_inv   = 1. / tsq;

        px += ((1.0 + tx * tx - ty * ty - tz * tz) * ux + 2.0 * (tx * ty + tz) * uy +
               2.0 * (tz * tx - ty) * uz) *
              tsq_inv;

        py += (2.0 * (tx * ty - tz) * ux + (1.0 - tx * tx + ty * ty - tz * tz) * uy +
               2.0 * (ty * tz + tx) * uz) *
              tsq_inv;

        pz += (2.0 * (tz * tx + ty) * ux + 2.0 * (ty * tz - tx) * uy +
               (1.0 - tx * tx - ty * ty + tz * tz) * uz) *
              tsq_inv;

        // gamma-factor
        usq   = (px * px + py * py + pz * pz);
        gamma = Kokkos::sqrt(1 + usq);

        // Update inverse gamma factor
        gamma_inv = 1 / gamma;

        // Update momentum
        mx(ip) = px;
        my(ip) = py;
        mz(ip) = pz;

        // Update positions
        x(ip) += mx(ip) * dt * gamma_inv;
        y(ip) += my(ip) * dt * gamma_inv;
        z(ip) += mz(ip) * dt * gamma_inv;
      });

    Kokkos::fence();

  } // Loop on species
}

//! \brief Push only the momentum.
//! \note Only used for the initialization of some setups.
//! \param[in] particles Vector of species Particles.
//! \param[in] dt Time step to use for the pusher.
void push_momentum(std::vector<Particles> &particles, double dt) {

  // for each species
  for (std::size_t is = 0; is < particles.size(); is++) {

    const std::size_t n_particles = particles[is].size();

    // q' = dt * (q/2m)
    const double qp = particles[is].charge_m * dt * 0.5 / particles[is].mass_m;

    Particles::view_t mx = particles[is].mx_m;
    Particles::view_t my = particles[is].my_m;
    Particles::view_t mz = particles[is].mz_m;

    Particles::view_t Exp = particles[is].Ex_m;
    Particles::view_t Eyp = particles[is].Ey_m;
    Particles::view_t Ezp = particles[is].Ez_m;

    Particles::view_t Bxp = particles[is].Bx_m;
    Particles::view_t Byp = particles[is].By_m;
    Particles::view_t Bzp = particles[is].Bz_m;

    Kokkos::parallel_for(
      n_particles,
      KOKKOS_LAMBDA(const int ip) {
        // 1/2 E
        double px = qp * Exp(ip);
        double py = qp * Eyp(ip);
        double pz = qp * Ezp(ip);

        const double ux = mx(ip) + px;
        const double uy = my(ip) + py;
        const double uz = mz(ip) + pz;

        // gamma-factor
        double usq       = (ux * ux + uy * uy + uz * uz);
        double gamma     = Kokkos::sqrt(1 + usq);
        double gamma_inv = qp / gamma;

        // B, T = Transform to rotate the particle
        const double tx  = gamma_inv * Bxp(ip);
        const double ty  = gamma_inv * Byp(ip);
        const double tz  = gamma_inv * Bzp(ip);
        const double tsq = 1. + (tx * tx + ty * ty + tz * tz);
        double tsq_inv   = 1. / tsq;

        px += ((1.0 + tx * tx - ty * ty - tz * tz) * ux + 2.0 * (tx * ty + tz) * uy +
               2.0 * (tz * tx - ty) * uz) *
              tsq_inv;

        py += (2.0 * (tx * ty - tz) * ux + (1.0 - tx * tx + ty * ty - tz * tz) * uy +
               2.0 * (ty * tz + tx) * uz) *
              tsq_inv;

        pz += (2.0 * (tz * tx + ty) * ux + 2.0 * (ty * tz - tx) * uy +
               (1.0 - tx * tx - ty * ty + tz * tz) * uz) *
              tsq_inv;

        // gamma-factor
        usq   = (px * px + py * py + pz * pz);
        gamma = Kokkos::sqrt(1 + usq);

        // Update inverse gamma factor
        gamma_inv = 1 / gamma;

        // Update momentum
        mx(ip) = px;
        my(ip) = py;
        mz(ip) = pz;
      });

    Kokkos::fence();
  } // end for species
}

//! \brief Boundaries condition on the particles, periodic
//! or reflect the particles which leave the domain.
//! \param[in] params Constant global simulation parameters.
//! \param[in] particles Vector of species particles.
void pushBC(const Params &params, std::vector<Particles> &particles) {
  const double inf_global[3] = {params.inf_x, params.inf_y, params.inf_z};
  const double sup_global[3] = {params.sup_x, params.sup_y, params.sup_z};

  // Periodic conditions
  if (params.boundary_condition_code == 1) {
    const double length[3] = {params.Lx, params.Ly, params.Lz};

    for (std::size_t is = 0; is < particles.size(); is++) {
      std::size_t n_particles = particles[is].size();

      Particles::view_t x = particles[is].x_m;
      Particles::view_t y = particles[is].y_m;
      Particles::view_t z = particles[is].z_m;

      Kokkos::parallel_for(
          n_particles,
          KOKKOS_LAMBDA(const int part) {
            double *pos[3] = {&x(part), &y(part), &z(part)};
            for (unsigned int d = 0; d < 3; d++) {
              if (*pos[d] >= sup_global[d]) {
                *pos[d] -= length[d];
              } else if (*pos[d] < inf_global[d]) {
                *pos[d] += length[d];
              }
            }
          } // End loop on particles
        );

      Kokkos::fence();

    } // End loop on species

    // Reflective conditions
  } else if (params.boundary_condition_code == 2) {
    for (std::size_t is = 0; is < particles.size(); is++) {
      std::size_t n_particles = particles[is].size();

      Particles::view_t x = particles[is].x_m;
      Particles::view_t y = particles[is].y_m;
      Particles::view_t z = particles[is].z_m;

      Particles::view_t mx = particles[is].mx_m;
      Particles::view_t my = particles[is].my_m;
      Particles::view_t mz = particles[is].mz_m;

      Kokkos::parallel_for(
          n_particles,
          KOKKOS_LAMBDA(const int part) {
            double *pos[3] = {&x(part), &y(part), &z(part)};
            double *momentum[3] = {&mx(part), &my(part), &mz(part)};

              for (int d = 0; d < 3; d++) {
                if (*pos[d] >= sup_global[d]) {
                  *pos[d]      = 2 * sup_global[d] - *pos[d];
                  *momentum[d] = -*momentum[d];
                } else if (*pos[d] < inf_global[d]) {
                  *pos[d]      = 2 * inf_global[d] - *pos[d];
                  *momentum[d] = -*momentum[d];
                }
              }
            } // End loop on particles
          );

      Kokkos::fence();

    } // End loop on species
  } // if type of conditions
}

//! \brief Current projection directly in the global array.
//! \param[in] params Constant global parameters.
//! \param[in] em Electromagnetic fields.
//! \param[in] particles Vector of species particles.
void project(const Params &params, ElectroMagn &em, std::vector<Particles> &particles) {
  ElectroMagn::view_t Jx_device = em.Jx_m;
  ElectroMagn::view_t Jy_device = em.Jy_m;
  ElectroMagn::view_t Jz_device = em.Jz_m;

#if defined(MINI_MINIPIC_KOKKOS_SCATTER_VIEW)
  // Use ScatterView
  Kokkos::Experimental::ScatterView<double ***> scatter_Jx(Jx_device);
  Kokkos::Experimental::ScatterView<double ***> scatter_Jy(Jy_device);
  Kokkos::Experimental::ScatterView<double ***> scatter_Jz(Jz_device);
#else
  // Use atomic memory traits
  Kokkos::View<double ***, Kokkos::MemoryTraits<Kokkos::Atomic>> Jx(Jx_device);
  Kokkos::View<double ***, Kokkos::MemoryTraits<Kokkos::Atomic>> Jy(Jy_device);
  Kokkos::View<double ***, Kokkos::MemoryTraits<Kokkos::Atomic>> Jz(Jz_device);
#endif

  const double dt = params.dt;

  const double inv_dx = params.inv_dx;
  const double inv_dy = params.inv_dy;
  const double inv_dz = params.inv_dz;

#if (MINI_MINIPIC_DEBUG)
  int nx_Jx = em.Jx_m.nx_m;
  int ny_Jx = em.Jx_m.ny_m;
  int nz_Jx = em.Jx_m.nz_m;

  int nx_Jy = em.Jy_m.nx_m;
  int ny_Jy = em.Jy_m.ny_m;
  int nz_Jy = em.Jy_m.nz_m;
#endif

  for (std::size_t is = 0; is < particles.size(); is++) {

    const std::size_t n_particles            = particles[is].size();
    const double inv_cell_volume_x_q = params.inv_cell_volume * particles[is].charge_m;

    Particles::view_t w = particles[is].weight_m;

    Particles::view_t x = particles[is].x_m;
    Particles::view_t y = particles[is].y_m;
    Particles::view_t z = particles[is].z_m;

    Particles::view_t mx = particles[is].mx_m;
    Particles::view_t my = particles[is].my_m;
    Particles::view_t mz = particles[is].mz_m;

    Kokkos::parallel_for(
      n_particles,
      KOKKOS_LAMBDA(const int part) {
#if defined(MINI_MINIPIC_KOKKOS_SCATTER_VIEW)
        auto Jx = scatter_Jx.access();
        auto Jy = scatter_Jy.access();
        auto Jz = scatter_Jz.access();
#endif
        // Delete if already compute by Pusher
        // double usq = (moment[0]*moment[0] + moment[1]*moment[1] + moment[2]*moment[2]);
        // double gamma = sqrt(1+usq);
        // gamma_inv = 1/gamma;

        const double charge_weight = inv_cell_volume_x_q * w(part);

        const double gamma_inv =
          1 / sqrt(1 + (mx(part) * mx(part) + my(part) * my(part) + mz(part) * mz(part)));

        const double vx = mx(part) * gamma_inv;
        const double vy = my(part) * gamma_inv;
        const double vz = mz(part) * gamma_inv;

        const double Jxp = vx * charge_weight;
        const double Jyp = vy * charge_weight;
        const double Jzp = vz * charge_weight;

        // Calculate normalized positions
        // We come back 1/2 time step back in time for the position because of the leap frog scheme
        // As a consequence, we also have `+ 1` because the current grids have 2 additional ghost
        // cells (1 the min and 1 at the max border) when the direction is primal
        const double posxn = (x(part) - 0.5 * dt * vx) * inv_dx + 1;
        const double posyn = (y(part) - 0.5 * dt * vy) * inv_dy + 1;
        const double poszn = (z(part) - 0.5 * dt * vz) * inv_dz + 1;

        // Compute indexes in primal grid
        const int ixp = (int)(floor(posxn));
        const int iyp = (int)(floor(posyn));
        const int izp = (int)(floor(poszn));

        // Compute indexes in dual grid
        const int ixd = (int)floor(posxn - 0.5);
        const int iyd = (int)floor(posyn - 0.5);
        const int izd = (int)floor(poszn - 0.5);

        // Projection particle on currant field
        // Compute interpolation coeff, p = primal, d = dual

        double coeffs[3];

        coeffs[0] = posxn - 0.5 - ixd;
        coeffs[1] = posyn - iyp;
        coeffs[2] = poszn - izp;

        Jx(ixd, iyp, izp) += (1 - coeffs[0]) * (1 - coeffs[1]) * (1 - coeffs[2]) * Jxp;
        Jx(ixd, iyp, izp + 1) += (1 - coeffs[0]) * (1 - coeffs[1]) * (coeffs[2]) * Jxp;
        Jx(ixd, iyp + 1, izp) += (1 - coeffs[0]) * (coeffs[1]) * (1 - coeffs[2]) * Jxp;
        Jx(ixd, iyp + 1, izp + 1) += (1 - coeffs[0]) * (coeffs[1]) * (coeffs[2]) * Jxp;
        Jx(ixd + 1, iyp, izp) += (coeffs[0]) * (1 - coeffs[1]) * (1 - coeffs[2]) * Jxp;
        Jx(ixd + 1, iyp, izp + 1) += (coeffs[0]) * (1 - coeffs[1]) * (coeffs[2]) * Jxp;
        Jx(ixd + 1, iyp + 1, izp) += (coeffs[0]) * (coeffs[1]) * (1 - coeffs[2]) * Jxp;
        Jx(ixd + 1, iyp + 1, izp + 1) += (coeffs[0]) * (coeffs[1]) * (coeffs[2]) * Jxp;

        coeffs[0] = posxn - ixp;
        coeffs[1] = posyn - 0.5 - iyd;
        coeffs[2] = poszn - izp;

        Jy(ixp, iyd, izp) += (1 - coeffs[0]) * (1 - coeffs[1]) * (1 - coeffs[2]) * Jyp;
        Jy(ixp, iyd, izp + 1) += (1 - coeffs[0]) * (1 - coeffs[1]) * (coeffs[2]) * Jyp;
        Jy(ixp, iyd + 1, izp) += (1 - coeffs[0]) * (coeffs[1]) * (1 - coeffs[2]) * Jyp;
        Jy(ixp, iyd + 1, izp + 1) += (1 - coeffs[0]) * (coeffs[1]) * (coeffs[2]) * Jyp;
        Jy(ixp + 1, iyd, izp) += (coeffs[0]) * (1 - coeffs[1]) * (1 - coeffs[2]) * Jyp;
        Jy(ixp + 1, iyd, izp + 1) += (coeffs[0]) * (1 - coeffs[1]) * (coeffs[2]) * Jyp;
        Jy(ixp + 1, iyd + 1, izp) += (coeffs[0]) * (coeffs[1]) * (1 - coeffs[2]) * Jyp;
        Jy(ixp + 1, iyd + 1, izp + 1) += (coeffs[0]) * (coeffs[1]) * (coeffs[2]) * Jyp;

        coeffs[0] = posxn - ixp;
        coeffs[1] = posyn - iyp;
        coeffs[2] = poszn - 0.5 - izd;

        Jz(ixp, iyp, izd) += (1 - coeffs[0]) * (1 - coeffs[1]) * (1 - coeffs[2]) * Jzp;
        Jz(ixp, iyp, izd + 1) += (1 - coeffs[0]) * (1 - coeffs[1]) * (coeffs[2]) * Jzp;
        Jz(ixp, iyp + 1, izd) += (1 - coeffs[0]) * (coeffs[1]) * (1 - coeffs[2]) * Jzp;
        Jz(ixp, iyp + 1, izd + 1) += (1 - coeffs[0]) * (coeffs[1]) * (coeffs[2]) * Jzp;
        Jz(ixp + 1, iyp, izd) += (coeffs[0]) * (1 - coeffs[1]) * (1 - coeffs[2]) * Jzp;
        Jz(ixp + 1, iyp, izd + 1) += (coeffs[0]) * (1 - coeffs[1]) * (coeffs[2]) * Jzp;
        Jz(ixp + 1, iyp + 1, izd) += (coeffs[0]) * (coeffs[1]) * (1 - coeffs[2]) * Jzp;
        Jz(ixp + 1, iyp + 1, izd + 1) += (coeffs[0]) * (coeffs[1]) * (coeffs[2]) * Jzp;

      }
    );  // end for each particles

    Kokkos::fence();
#if defined(MINI_MINIPIC_KOKKOS_SCATTER_VIEW)
    Kokkos::Experimental::contribute(Jx_device, scatter_Jx);
    Kokkos::Experimental::contribute(Jy_device, scatter_Jy);
    Kokkos::Experimental::contribute(Jz_device, scatter_Jz);
#endif
  } // end for each species
}

//! \brief Solve Maxwell equations to compute EM fields.
//! \param[in] params Constant global parameters.
//! \param[in] em Electromagnetic fields.
void solve_maxwell(const Params &params, ElectroMagn &em) {
  const double dt         = params.dt;
  const double dt_over_dx = params.dt * params.inv_dx;
  const double dt_over_dy = params.dt * params.inv_dy;
  const double dt_over_dz = params.dt * params.inv_dz;

  /////     Solve Maxwell Ampere (E)
  // Electric field Ex (d,p,p)

  ElectroMagn::view_t Jx = em.Jx_m;
  ElectroMagn::view_t Jy = em.Jy_m;
  ElectroMagn::view_t Jz = em.Jz_m;

  ElectroMagn::view_t Ex = em.Ex_m;
  ElectroMagn::view_t Ey = em.Ey_m;
  ElectroMagn::view_t Ez = em.Ez_m;

  ElectroMagn::view_t Bx = em.Bx_m;
  ElectroMagn::view_t By = em.By_m;
  ElectroMagn::view_t Bz = em.Bz_m;

  typedef Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<3>> mdrange_policy;

  // Electric field Ex (d,p,p)
  Kokkos::parallel_for(
    mdrange_policy({0, 0, 0}, {em.nx_d_m, em.ny_p_m, em.nz_p_m}),
    KOKKOS_LAMBDA(const int ix, const int iy, const int iz) {
      Ex(ix, iy, iz) += -dt * Jx(ix, iy + 1, iz + 1) +
                        dt_over_dy * (Bz(ix, iy + 1, iz) - Bz(ix, iy, iz)) -
                        dt_over_dz * (By(ix, iy, iz + 1) - By(ix, iy, iz));
    });

  // Electric field Ey (p,d,p)
  Kokkos::parallel_for(
    mdrange_policy({0, 0, 0}, {em.nx_p_m, em.ny_d_m, em.nz_p_m}),
    KOKKOS_LAMBDA(const int ix, const int iy, const int iz) {
      Ey(ix, iy, iz) += -dt * Jy(ix + 1, iy, iz + 1) -
                        dt_over_dx * (Bz(ix + 1, iy, iz) - Bz(ix, iy, iz)) +
                        dt_over_dz * (Bx(ix, iy, iz + 1) - Bx(ix, iy, iz));
    });

  // Electric field Ez (p,p,d)

  Kokkos::parallel_for(
    mdrange_policy({0, 0, 0}, {em.nx_p_m, em.ny_p_m, em.nz_d_m}),
    KOKKOS_LAMBDA(const int ix, const int iy, const int iz) {
      Ez(ix, iy, iz) += -dt * Jz(ix + 1, iy + 1, iz) +
                        dt_over_dx * (By(ix + 1, iy, iz) - By(ix, iy, iz)) -
                        dt_over_dy * (Bx(ix, iy + 1, iz) - Bx(ix, iy, iz));
    });

  Kokkos::fence();

  /////     Solve Maxwell Faraday (B)

  // Magnetic field Bx (p,d,d)

  Kokkos::parallel_for(
    mdrange_policy({0, 1, 1}, {em.nx_p_m, em.ny_d_m - 1, em.nz_d_m - 1}),
    KOKKOS_LAMBDA(const int ix, const int iy, const int iz) {
      Bx(ix, iy, iz) += -dt_over_dy * (Ez(ix, iy, iz) - Ez(ix, iy - 1, iz)) +
                        dt_over_dz * (Ey(ix, iy, iz) - Ey(ix, iy, iz - 1));
    });

  // Magnetic field By (d,p,d)

  Kokkos::parallel_for(
    mdrange_policy({1, 0, 1}, {em.nx_d_m - 1, em.ny_p_m, em.nz_d_m - 1}),
    KOKKOS_LAMBDA(const int ix, const int iy, const int iz) {
      By(ix, iy, iz) += -dt_over_dz * (Ex(ix, iy, iz) - Ex(ix, iy, iz - 1)) +
                        dt_over_dx * (Ez(ix, iy, iz) - Ez(ix - 1, iy, iz));
    });

  // Magnetic field Bz (d,d,p)

  Kokkos::parallel_for(
    mdrange_policy({1, 1, 0}, {em.nx_d_m - 1, em.ny_d_m - 1, em.nz_p_m}),
    KOKKOS_LAMBDA(const int ix, const int iy, const int iz) {
      Bz(ix, iy, iz) += -dt_over_dx * (Ey(ix, iy, iz) - Ey(ix - 1, iy, iz)) +
                        dt_over_dy * (Ex(ix, iy, iz) - Ex(ix, iy - 1, iz));
    });

  Kokkos::fence();

} // end solve

//! \brief Boundaries condition on the global grid.
//! \param[in] params Global constant parameters.
//! \param[in] em Electromagnetic fields.
void currentBC(const Params &params, ElectroMagn &em) {

  if (params.boundary_condition == "periodic") {

    ElectroMagn::view_t Jx = em.Jx_m;
    ElectroMagn::view_t Jy = em.Jy_m;
    ElectroMagn::view_t Jz = em.Jz_m;

    const auto nx_Jx = em.Jx_m.extent(0);
    const auto ny_Jx = em.Jx_m.extent(1);
    const auto nz_Jx = em.Jx_m.extent(2);

    const auto nx_Jy = em.Jy_m.extent(0);
    const auto ny_Jy = em.Jy_m.extent(1);
    const auto nz_Jy = em.Jy_m.extent(2);

    const auto nx_Jz = em.Jz_m.extent(0);
    const auto ny_Jz = em.Jz_m.extent(1);
    const auto nz_Jz = em.Jz_m.extent(2);

    typedef Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<2>> mdrange_policy;

    // X

    Kokkos::parallel_for(
      mdrange_policy({0, 0}, {ny_Jx, nz_Jx}),
      KOKKOS_LAMBDA(const int iy, const int iz) {
        Jx(0, iy, iz) += Jx(nx_Jx - 2, iy, iz);
        Jx(nx_Jx - 2, iy, iz) = Jx(0, iy, iz);

        Jx(1, iy, iz) += Jx(nx_Jx - 1, iy, iz);
        Jx(nx_Jx - 1, iy, iz) = Jx(1, iy, iz);
      });

    Kokkos::parallel_for(
      mdrange_policy({0, 0}, {ny_Jy, nz_Jy}),
      KOKKOS_LAMBDA(const int iy, const int iz) {
        Jy(0, iy, iz) += Jy(nx_Jy - 2, iy, iz);
        Jy(nx_Jy - 2, iy, iz) = Jy(0, iy, iz);

        Jy(1, iy, iz) += Jy(nx_Jy - 1, iy, iz);
        Jy(nx_Jy - 1, iy, iz) = Jy(1, iy, iz);
      });

    Kokkos::parallel_for(
      mdrange_policy({0, 0}, {ny_Jz, nz_Jz}),
      KOKKOS_LAMBDA(const int iy, const int iz) {
        Jz(0, iy, iz) += Jz(nx_Jz - 2, iy, iz);
        Jz(nx_Jz - 2, iy, iz) = Jz(0, iy, iz);

        Jz(1, iy, iz) += Jz(nx_Jz - 1, iy, iz);
        Jz(nx_Jz - 1, iy, iz) = Jz(1, iy, iz);
      });

    Kokkos::fence();

    // Y

    Kokkos::parallel_for(
      mdrange_policy({0, 0}, {nx_Jx, nz_Jx}),
      KOKKOS_LAMBDA(const int ix, const int iz) {
        Jx(ix, 0, iz) += Jx(ix, ny_Jx - 2, iz);
        Jx(ix, ny_Jx - 2, iz) = Jx(ix, 0, iz);

        Jx(ix, 1, iz) += Jx(ix, ny_Jx - 1, iz);
        Jx(ix, ny_Jx - 1, iz) = Jx(ix, 1, iz);
      });

    Kokkos::parallel_for(
      mdrange_policy({0, 0}, {nx_Jy, nz_Jy}),
      KOKKOS_LAMBDA(const int ix, const int iz) {
        Jy(ix, 0, iz) += Jy(ix, ny_Jy - 2, iz);
        Jy(ix, ny_Jy - 2, iz) = Jy(ix, 0, iz);

        Jy(ix, 1, iz) += Jy(ix, ny_Jy - 1, iz);
        Jy(ix, ny_Jy - 1, iz) = Jy(ix, 1, iz);
      });

    Kokkos::parallel_for(
      mdrange_policy({0, 0}, {nx_Jz, nz_Jz}),
      KOKKOS_LAMBDA(const int ix, const int iz) {
        Jz(ix, 0, iz) += Jz(ix, ny_Jz - 2, iz);
        Jz(ix, ny_Jz - 2, iz) = Jz(ix, 0, iz);

        Jz(ix, 1, iz) += Jz(ix, ny_Jz - 1, iz);
        Jz(ix, ny_Jz - 1, iz) = Jz(ix, 1, iz);
      });

    Kokkos::fence();

    // Z

    Kokkos::parallel_for(
      mdrange_policy({0, 0}, {nx_Jx, ny_Jx}),
      KOKKOS_LAMBDA(const int ix, const int iy) {
        Jx(ix, iy, 0) += Jx(ix, iy, nz_Jx - 2);
        Jx(ix, iy, nz_Jx - 2) = Jx(ix, iy, 0);

        Jx(ix, iy, 1) += Jx(ix, iy, nz_Jx - 1);
        Jx(ix, iy, nz_Jx - 1) = Jx(ix, iy, 1);
      });

    Kokkos::parallel_for(
      mdrange_policy({0, 0}, {nx_Jy, ny_Jy}),
      KOKKOS_LAMBDA(const int ix, const int iy) {
        Jy(ix, iy, 0) += Jy(ix, iy, nz_Jy - 2);
        Jy(ix, iy, nz_Jy - 2) = Jy(ix, iy, 0);

        Jy(ix, iy, 1) += Jy(ix, iy, nz_Jy - 1);
        Jy(ix, iy, nz_Jy - 1) = Jy(ix, iy, 1);
      });

    Kokkos::parallel_for(
      mdrange_policy({0, 0}, {nx_Jz, ny_Jz}),
      KOKKOS_LAMBDA(const int ix, const int iy) {
        Jz(ix, iy, 0) += Jz(ix, iy, nz_Jz - 2);
        Jz(ix, iy, nz_Jz - 2) = Jz(ix, iy, 0);

        Jz(ix, iy, 1) += Jz(ix, iy, nz_Jz - 1);
        Jz(ix, iy, nz_Jz - 1) = Jz(ix, iy, 1);
      });

    Kokkos::fence();

  } // end if periodic
} // end currentBC

//! \brief Boundaries condition on the global grid.
//! \param[in] params Global constant parameters.
//! \param[in] em Electromagnetic fields.
void solveBC(const Params &params, ElectroMagn &em) {
  typedef Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<2>> mdrange_policy;

  if (params.boundary_condition == "periodic") {

    ElectroMagn::view_t Bx = em.Bx_m;
    ElectroMagn::view_t By = em.By_m;
    ElectroMagn::view_t Bz = em.Bz_m;

    const auto nx_Bx = em.Bx_m.extent(0);
    const auto ny_Bx = em.Bx_m.extent(1);
    const auto nz_Bx = em.Bx_m.extent(2);

    const auto nx_By = em.By_m.extent(0);
    const auto ny_By = em.By_m.extent(1);
    const auto nz_By = em.By_m.extent(2);

    const auto nx_Bz = em.Bz_m.extent(0);
    const auto ny_Bz = em.Bz_m.extent(1);
    const auto nz_Bz = em.Bz_m.extent(2);

    // X dim
    // By (d,p,d)

    Kokkos::parallel_for(
      mdrange_policy({0, 0}, {ny_By, nz_By}),
      KOKKOS_LAMBDA(const int iy, const int iz) {
        // -X
        By(0, iy, iz)         = By(nx_By - 2, iy, iz);
        By(nx_By - 1, iy, iz) = By(1, iy, iz);
      });

    // Bz (d,d,p)
    Kokkos::parallel_for(
      mdrange_policy({0, 0}, {ny_Bz, nz_Bz}),
      KOKKOS_LAMBDA(const int iy, const int iz) {
        // -X
        Bz(0, iy, iz)         = Bz(nx_Bz - 2, iy, iz);
        Bz(nx_Bz - 1, iy, iz) = Bz(1, iy, iz);
      });

    Kokkos::fence();

    // Y dim
    // Bx (p,d,d)

    Kokkos::parallel_for(
      mdrange_policy({0, 0}, {nx_Bx, nz_Bx}),
      KOKKOS_LAMBDA(const int ix, const int iz) {
        // -Y
        Bx(ix, 0, iz) = Bx(ix, ny_Bx - 2, iz);
        // +Y
        Bx(ix, ny_Bx - 1, iz) = Bx(ix, 1, iz);
      });

    // Bz (d,d,p)

    Kokkos::parallel_for(
      mdrange_policy({0, 0}, {nx_Bz, nz_Bz}),
      KOKKOS_LAMBDA(const int ix, const int iz) {
        // -Y
        Bz(ix, 0, iz) = Bz(ix, ny_Bz - 2, iz);
        // +Y
        Bz(ix, ny_Bz - 1, iz) = Bz(ix, 1, iz);
      });

    Kokkos::fence();

    // Z dim
    // Bx

    Kokkos::parallel_for(
      mdrange_policy({0, 0}, {nx_Bx, ny_Bx}),
      KOKKOS_LAMBDA(const int ix, const int iy) {
        // -Z
        Bx(ix, iy, 0) = Bx(ix, iy, nz_Bx - 2);
        // +Z
        Bx(ix, iy, nz_Bx - 1) = Bx(ix, iy, 1);
      });

    // By
    Kokkos::parallel_for(
      mdrange_policy({0, 0}, {nx_By, ny_By}),
      KOKKOS_LAMBDA(const int ix, const int iy) {
        // -Z
        By(ix, iy, 0) = By(ix, iy, nz_By - 2);
        // +Z
        By(ix, iy, nz_By - 1) = By(ix, iy, 1);
      });

    Kokkos::fence();

  } else if (params.boundary_condition == "reflective") {

    ElectroMagn::view_t Bx = em.Bx_m;
    ElectroMagn::view_t By = em.By_m;
    ElectroMagn::view_t Bz = em.Bz_m;

    const auto nx_Bx = em.Bx_m.extent(0);
    const auto ny_Bx = em.Bx_m.extent(1);
    const auto nz_Bx = em.Bx_m.extent(2);

    const auto nx_By = em.By_m.extent(0);
    const auto ny_By = em.By_m.extent(1);
    const auto nz_By = em.By_m.extent(2);

    const auto nx_Bz = em.Bz_m.extent(0);
    const auto ny_Bz = em.Bz_m.extent(1);
    const auto nz_Bz = em.Bz_m.extent(2);

    // X dim
    // By (d,p,d)
    Kokkos::parallel_for(
      mdrange_policy({0, 0}, {ny_By, nz_By}),
      KOKKOS_LAMBDA(const int iy, const int iz) {
        // -X
        By(0, iy, iz) = By(1, iy, iz);
        // +X
        By(nx_By - 1, iy, iz) = By(nx_By - 2, iy, iz);
      });

    // Bz (d,d,p)
    Kokkos::parallel_for(
      mdrange_policy({0, 0}, {ny_Bz, nz_Bz}),
      KOKKOS_LAMBDA(const int iy, const int iz) {
        // -X
        Bz(0, iy, iz) = Bz(1, iy, iz);
        // +X
        Bz(nx_Bz - 1, iy, iz) = Bz(nx_Bz - 2, iy, iz);
      });

    // Y dim
    // Bx (p,d,d)
    Kokkos::parallel_for(
      mdrange_policy({0, 0}, {nx_Bx, nz_Bx}),
      KOKKOS_LAMBDA(const int ix, const int iz) {
        // -Y
        Bx(ix, 0, iz) = Bx(ix, 1, iz);
        // +Y
        Bx(ix, ny_Bx - 1, iz) = Bx(ix, ny_Bx - 2, iz);
      });

    Kokkos::fence();

    // Bz (-1 to avoid corner)
    Kokkos::parallel_for(
      mdrange_policy({0, 0}, {nx_Bz, nz_Bz}),
      KOKKOS_LAMBDA(const int ix, const int iz) {
        // -Y
        Bz(ix, 0, iz) = Bz(ix, 1, iz);
        // +Y
        Bz(ix, ny_Bz - 1, iz) = Bz(ix, ny_Bz - 2, iz);
      });

    // Z dim
    // Bx
    Kokkos::parallel_for(
      mdrange_policy({0, 0}, {nx_Bx, ny_Bx}),
      KOKKOS_LAMBDA(const int ix, const int iy) {
        // -Z
        Bx(ix, iy, 0) = Bx(ix, iy, 1);
        // +Z
        Bx(ix, iy, nz_Bx - 1) = Bx(ix, iy, nz_Bx - 2);
      });

    // By
    Kokkos::parallel_for(
      mdrange_policy({0, 0}, {nx_By, ny_By}),
      KOKKOS_LAMBDA(const int ix, const int iy) {
        // -Z
        By(ix, iy, 0) = By(ix, iy, 1);
        // +Z
        By(ix, iy, nz_By - 1) = By(ix, iy, nz_By - 2);
      });

    Kokkos::fence();
  } // End if
} // End solveBC

//! \brief Emit a laser field in the $x$ direction using an antenna.
//! \note Only used for some setups.
//! \param[in] params Global constant parameters.
//! \param[in] em Electromagnetic fields.
//! \param[in] profile Profile of the antenna.
//! \param[in] x Position of the antenna.
//! \param[in] t Current time.
void antenna(const Params &params,
             ElectroMagn &em,
             std::function<double(double, double, double)> profile,
             double x,
             double t) {

  em.sync(minipic::device, minipic::host);

  ElectroMagn::hostview_t *J = &em.Jz_h_m;

  const int ix = Kokkos::floor((x - params.inf_x - em.J_dual_zx_m * 0.5 * params.dx) / params.dx);

  const double yfs = 0.5 * params.Ly + params.inf_y;
  const double zfs = 0.5 * params.Lz + params.inf_z;

  for (std::size_t iy = 0; iy < J->extent(1); ++iy) {
    for (std::size_t iz = 0; iz < J->extent(2); ++iz) {

      const double y = (iy - em.J_dual_zy_m * 0.5) * params.dy + params.inf_y - yfs;
      const double z = (iz - em.J_dual_zz_m * 0.5) * params.dz + params.inf_z - zfs;

      (*J)(ix, iy, iz) = profile(y, z, t);
    }
  }

  em.sync(minipic::host, minipic::device);

} // end antenna

} // end namespace operators

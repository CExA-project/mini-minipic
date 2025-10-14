/* _____________________________________________________________________ */
//! \file Operators.hpp

//! \brief contains generic kernels for the particle pusher

/* _____________________________________________________________________ */

#ifndef OPERATORS_H
#define OPERATORS_H

namespace operators {

// ______________________________________________________________________________
//
//! \brief Interpolation operator at the patch level :
//! interpolate EM fields from global grid for each particle
//! \param[in] em  global electromagnetic fields
//! \param[in] patch  patch data structure
// ______________________________________________________________________________
auto interpolate(ElectroMagn &em, Patch &patch) -> void {

  const auto inv_dx_m = em.inv_dx_m;
  const auto inv_dy_m = em.inv_dy_m;
  const auto inv_dz_m = em.inv_dz_m;

  // em.Ex_m.print();

  for (int is = 0; is < patch.n_species_m; is++) {

    const int n_particles = patch.particles_m[is].size();

    device_field_t Ex = em.Ex_m.data_m;
    device_field_t Ey = em.Ey_m.data_m;
    device_field_t Ez = em.Ez_m.data_m;

    device_field_t Bx = em.Bx_m.data_m;
    device_field_t By = em.By_m.data_m;
    device_field_t Bz = em.Bz_m.data_m;

    device_vector_t x = patch.particles_m[is].x_.data_;
    device_vector_t y = patch.particles_m[is].y_.data_;
    device_vector_t z = patch.particles_m[is].z_.data_;

    device_vector_t Exp = patch.particles_m[is].Ex_.data_;
    device_vector_t Eyp = patch.particles_m[is].Ey_.data_;
    device_vector_t Ezp = patch.particles_m[is].Ez_.data_;

    device_vector_t Bxp = patch.particles_m[is].Bx_.data_;
    device_vector_t Byp = patch.particles_m[is].By_.data_;
    device_vector_t Bzp = patch.particles_m[is].Bz_.data_;

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

        double coeffs[3] = {ixn + 0.5, iyn, izn};

        // interpolation electric field
        // Ex (d, p , p)
        // {
        //   const double coeffs[3] = {ixn + 0.5, iyn, izn};
        //   Exp(part)              = compute_interpolation(coeffs,
        //                                     Ex(ixd, iyp, izp),
        //                                     Ex(ixd, iyp, izp + 1),
        //                                     Ex(ixd, iyp + 1, izp),
        //                                     Ex(ixd, iyp + 1, izp + 1),
        //                                     Ex(ixd + 1, iyp, izp),
        //                                     Ex(ixd + 1, iyp, izp + 1),
        //                                     Ex(ixd + 1, iyp + 1, izp),
        //                                     Ex(ixd + 1, iyp + 1, izp + 1));
        // }
        {
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
          //   Eyp(part)              = compute_interpolation(coeffs,
          //                                     Ey(ixp, iyd, izp),
          //                                     Ey(ixp, iyd, izp + 1),
          //                                     Ey(ixp, iyd + 1, izp),
          //                                     Ey(ixp, iyd + 1, izp + 1),
          //                                     Ey(ixp + 1, iyd, izp),
          //                                     Ey(ixp + 1, iyd, izp + 1),
          //                                     Ey(ixp + 1, iyd + 1, izp),
          //                                     Ey(ixp + 1, iyd + 1, izp + 1));

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

        // // //particles_m[is].Ey_.d_view(part) = compute_interpolation(ixp, b, izp, coeffs, Ey);
        // Ez (p, p, d)
        {
          const double coeffs[3] = {ixn, iyn, izn + 0.5};
          //   Ezp(part)              = compute_interpolation(coeffs,
          //                                     Ez(ixp, iyp, izd),
          //                                     Ez(ixp, iyp, izd + 1),
          //                                     Ez(ixp, iyp + 1, izd),
          //                                     Ez(ixp, iyp + 1, izd + 1),
          //                                     Ez(ixp + 1, iyp, izd),
          //                                     Ez(ixp + 1, iyp, izd + 1),
          //                                     Ez(ixp + 1, iyp + 1, izd),
          //                                     Ez(ixp + 1, iyp + 1, izd + 1));

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
        // particles_m[is].Ez_.d_view(part) = compute_interpolation(ixp, iyp, g, coeffs, Ez);

        // interpolation magnetic field
        // Bx (p, d, d)
        {
          const double coeffs[3] = {ixn, iyn + 0.5, izn + 0.5};

          // Bxp(part)              = compute_interpolation(coeffs,
          //                                   Bx(ixp, iyd, izd),
          //                                   Bx(ixp, iyd, izd + 1),
          //                                   Bx(ixp, iyd + 1, izd),
          //                                   Bx(ixp, iyd + 1, izd + 1),
          //                                   Bx(ixp + 1, iyd, izd),
          //                                   Bx(ixp + 1, iyd, izd + 1),
          //                                   Bx(ixp + 1, iyd + 1, izd),
          //                                   Bx(ixp + 1, iyd + 1, izd + 1));

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
        // particles_m[is].Bx_.d_view(part) = compute_interpolation(ixp, b, g, coeffs, Bx);

        // By (d, p, d)
        {
          const double coeffs[3] = {ixn + 0.5, iyn, izn + 0.5};

          // Byp(part)              = compute_interpolation(coeffs,
          //                                   By(ixd, iyp, izd),
          //                                   By(ixd, iyp, izd + 1),
          //                                   By(ixd, iyp + 1, izd),
          //                                   By(ixd, iyp + 1, izd + 1),
          //                                   By(ixd + 1, iyp, izd),
          //                                   By(ixd + 1, iyp, izd + 1),
          //                                   By(ixd + 1, iyp + 1, izd),
          //                                   By(ixd + 1, iyp + 1, izd + 1));

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
        // particles_m[is].By_.d_view(part) = compute_interpolation(a, iyp, g, coeffs, By);

        // Bz (d, d, p)
        {
          const double coeffs[3] = {ixn + 0.5, iyn + 0.5, izn};

          // Bzp(part)              = compute_interpolation(coeffs,
          //                                   Bz(ixd, iyd, izp),
          //                                   Bz(ixd, iyd, izp + 1),
          //                                   Bz(ixd, iyd + 1, izp),
          //                                   Bz(ixd, iyd + 1, izp + 1),
          //                                   Bz(ixd + 1, iyd, izp),
          //                                   Bz(ixd + 1, iyd, izp + 1),
          //                                   Bz(ixd + 1, iyd + 1, izp),
          //                                   Bz(ixd + 1, iyd + 1, izp + 1));

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
        // particles_m[is].Bz_.d_view(part) = compute_interpolation(a, b, izp, coeffs, Bz);
      } // End for each particle

    ); // end KOKKOS PARALLEL

    Kokkos::fence();

  } // Species loop
}

// ______________________________________________________________________________
//
//! \brief Move the particle in the space, compute with EM fields interpolate
//! \param[in] patch  patch data structure
//! \param[in] dt time step to use for the pusher
// ______________________________________________________________________________
auto push(Patch &patch, double dt) -> void {

  // For each species
  for (int is = 0; is < patch.n_species_m; is++) {

    const int n_particles = patch.particles_m[is].size();

    // q' = dt * (q/2m)
    const mini_float qp = patch.particles_m[is].charge_m * dt * 0.5 / patch.particles_m[is].mass_m;

    device_vector_t x = patch.particles_m[is].x_.data_;
    device_vector_t y = patch.particles_m[is].y_.data_;
    device_vector_t z = patch.particles_m[is].z_.data_;

    device_vector_t mx = patch.particles_m[is].mx_.data_;
    device_vector_t my = patch.particles_m[is].my_.data_;
    device_vector_t mz = patch.particles_m[is].mz_.data_;

    device_vector_t Exp = patch.particles_m[is].Ex_.data_;
    device_vector_t Eyp = patch.particles_m[is].Ey_.data_;
    device_vector_t Ezp = patch.particles_m[is].Ez_.data_;

    device_vector_t Bxp = patch.particles_m[is].Bx_.data_;
    device_vector_t Byp = patch.particles_m[is].By_.data_;
    device_vector_t Bzp = patch.particles_m[is].Bz_.data_;

    Kokkos::parallel_for(
      n_particles,
      KOKKOS_LAMBDA(const int ip) {
        // 1/2 E
        mini_float px = qp * Exp(ip);
        mini_float py = qp * Eyp(ip);
        mini_float pz = qp * Ezp(ip);

        const mini_float ux = mx(ip) + px;
        const mini_float uy = my(ip) + py;
        const mini_float uz = mz(ip) + pz;

        // gamma-factor
        mini_float usq       = (ux * ux + uy * uy + uz * uz);
        mini_float gamma     = Kokkos::sqrt(1 + usq);
        mini_float gamma_inv = qp / gamma;

        // B, T = Transform to rotate the particle
        const mini_float tx  = gamma_inv * Bxp(ip);
        const mini_float ty  = gamma_inv * Byp(ip);
        const mini_float tz  = gamma_inv * Bzp(ip);
        const mini_float tsq = 1. + (tx * tx + ty * ty + tz * tz);
        mini_float tsq_inv   = 1. / tsq;

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

    // Copy the data from the device to the host
    // patch.particles_m[is].copy_device_to_host();

    // patch.particles_m[is].check(patch.inf_m[0], patch.sup_m[0], patch.inf_m[1], patch.sup_m[1],
    // patch.inf_m[2], patch.sup_m[2]);

    // patch.particles_m[is].print();

    // patch.particles_m[is].check_sum();

  } // Loop on species
}

// ______________________________________________________________________________
//
//! \brief Push only the momentum
//! \param[in] patch  patch data structure
//! \param[in] dt time step to use for the pusher
// ______________________________________________________________________________
auto push_momentum(Patch &patch, double dt) -> void {

  // for each species
  for (int is = 0; is < patch.n_species_m; is++) {

    const int n_particles = patch.particles_m[is].size();

    // q' = dt * (q/2m)
    const mini_float qp = patch.particles_m[is].charge_m * dt * 0.5 / patch.particles_m[is].mass_m;

    device_vector_t mx = patch.particles_m[is].mx_.data_;
    device_vector_t my = patch.particles_m[is].my_.data_;
    device_vector_t mz = patch.particles_m[is].mz_.data_;

    device_vector_t Exp = patch.particles_m[is].Ex_.data_;
    device_vector_t Eyp = patch.particles_m[is].Ey_.data_;
    device_vector_t Ezp = patch.particles_m[is].Ez_.data_;

    device_vector_t Bxp = patch.particles_m[is].Bx_.data_;
    device_vector_t Byp = patch.particles_m[is].By_.data_;
    device_vector_t Bzp = patch.particles_m[is].Bz_.data_;

    Kokkos::parallel_for(
      n_particles,
      KOKKOS_LAMBDA(const int ip) {
        // 1/2 E
        mini_float px = qp * Exp(ip);
        mini_float py = qp * Eyp(ip);
        mini_float pz = qp * Ezp(ip);

        const mini_float ux = mx(ip) + px;
        const mini_float uy = my(ip) + py;
        const mini_float uz = mz(ip) + pz;

        // gamma-factor
        mini_float usq       = (ux * ux + uy * uy + uz * uz);
        mini_float gamma     = Kokkos::sqrt(1 + usq);
        mini_float gamma_inv = qp / gamma;

        // B, T = Transform to rotate the particle
        const mini_float tx  = gamma_inv * Bxp(ip);
        const mini_float ty  = gamma_inv * Byp(ip);
        const mini_float tz  = gamma_inv * Bzp(ip);
        const mini_float tsq = 1. + (tx * tx + ty * ty + tz * tz);
        mini_float tsq_inv   = 1. / tsq;

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

// _____________________________________________________________________
//
//! \brief Boundaries condition on the particles, periodic
//! or reflect the particles which leave the domain
//
//! \param[in] Params & params - constant global simulation parameters
//! \param[in] Patch & patch - current patch
// _____________________________________________________________________
auto pushBC(Params &params, Patch &patch) -> void {

  if (patch.on_border_m) {

    const mini_float inf_global[3] = {params.inf_x, params.inf_y, params.inf_z};
    const mini_float sup_global[3] = {params.sup_x, params.sup_y, params.sup_z};

    // Periodic conditions
    if (params.boundary_condition_code == 1) {

      const int N_patches[3]     = {patch.nx_patchs_m, patch.ny_patchs_m, patch.nz_patchs_m};
      const mini_float length[3] = {params.Lx, params.Ly, params.Lz};

      for (int is = 0; is < patch.n_species_m; is++) {

        unsigned int n_particles = patch.particles_m[is].size();

        device_vector_t x = patch.particles_m[is].x_.data_;
        device_vector_t y = patch.particles_m[is].y_.data_;
        device_vector_t z = patch.particles_m[is].z_.data_;

        Kokkos::parallel_for(
          n_particles,
          KOKKOS_LAMBDA(const int part) {
            mini_float *pos[3] = {&x(part), &y(part), &z(part)};

            for (int d = 0; d < 3; d++) {

              // Only relevant if there is just 1 patch in this direction
              // Else the patch exchange with periodicity is managed in the dedicated function
              if (N_patches[d] == 1) {
                if (*pos[d] >= sup_global[d]) {

                  *pos[d] -= length[d];

                } else if (*pos[d] < inf_global[d]) {

                  *pos[d] += length[d];
                }
              }
            }
          } // End loop on particles

        );

        Kokkos::fence();

      } // End loop on species

      // Reflective conditions
    } else if (params.boundary_condition_code == 2) {
      for (int is = 0; is < patch.n_species_m; is++) {

        unsigned int n_particles = patch.particles_m[is].size();

        device_vector_t x = patch.particles_m[is].x_.data_;
        device_vector_t y = patch.particles_m[is].y_.data_;
        device_vector_t z = patch.particles_m[is].z_.data_;

        device_vector_t mx = patch.particles_m[is].mx_.data_;
        device_vector_t my = patch.particles_m[is].my_.data_;
        device_vector_t mz = patch.particles_m[is].mz_.data_;

        Kokkos::parallel_for(
          n_particles,
          KOKKOS_LAMBDA(const int part) {
            mini_float *pos[3] = {&x(part), &y(part), &z(part)};

            mini_float *momentum[3] = {&mx(part), &my(part), &mz(part)};

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
  } // if on border
}

// _______________________________________________________________________
//
//! \brief Current projection from global particles position to local grid
//! \param params global parameters
//! \param patch current patch
// _______________________________________________________________________
auto project(Params &params, Patch &patch) -> void {

  for (int is = 0; is < patch.n_species_m; is++) {

    patch.vec_Jx_m[is].reset(minipic::device);
    patch.vec_Jy_m[is].reset(minipic::device);
    patch.vec_Jz_m[is].reset(minipic::device);

#if defined(__MINIPIC_DEBUG__)
    patch.particles_m[is].sync(minipic::device, minipic::host);
    patch.particles_m[is].check(params.inf_x - params.dx,
                                params.sup_x + params.dx,
                                params.inf_y - params.dy,
                                params.sup_y + params.dy,
                                params.inf_z - params.dz,
                                params.sup_z + params.dz);
    // particles_m[is].print();
    // particles_m[is].check_sum();
#endif

    const int n_particles = patch.particles_m[is].size();
    if (n_particles > 0) {

      const double inv_cell_volume_x_q = params.inv_cell_volume * patch.particles_m[is].charge_m;
      const double dt                  = params.dt;

      const double inv_dx = params.inv_dx;
      const double inv_dy = params.inv_dy;
      const double inv_dz = params.inv_dz;

      const double xmin = patch.inf_m[0];
      const double ymin = patch.inf_m[1];
      const double zmin = patch.inf_m[2];

      device_field_t Jx_species = patch.vec_Jx_m[is].data_m;
      device_field_t Jy_species = patch.vec_Jy_m[is].data_m;
      device_field_t Jz_species = patch.vec_Jz_m[is].data_m;

      device_vector_t w = patch.particles_m[is].weight_.data_;

      device_vector_t x = patch.particles_m[is].x_.data_;
      device_vector_t y = patch.particles_m[is].y_.data_;
      device_vector_t z = patch.particles_m[is].z_.data_;

      device_vector_t mx = patch.particles_m[is].mx_.data_;
      device_vector_t my = patch.particles_m[is].my_.data_;
      device_vector_t mz = patch.particles_m[is].mz_.data_;

#if defined(__MINIPIC_KOKKOS_SCATTERVIEW__)
      Kokkos::Experimental::ScatterView<double ***> scatter_Jx_loc(Jx_species);
      Kokkos::Experimental::ScatterView<double ***> scatter_Jy_loc(Jy_species);
      Kokkos::Experimental::ScatterView<double ***> scatter_Jz_loc(Jz_species);
#else
      Kokkos::View<double ***, Kokkos::MemoryTraits<Kokkos::Atomic>> Jx_loc = Jx_species;
      Kokkos::View<double ***, Kokkos::MemoryTraits<Kokkos::Atomic>> Jy_loc = Jy_species;
      Kokkos::View<double ***, Kokkos::MemoryTraits<Kokkos::Atomic>> Jz_loc = Jz_species;
#endif

      Kokkos::parallel_for(
        n_particles,
        KOKKOS_LAMBDA(const int part) {

#if defined(__MINIPIC_KOKKOS_SCATTERVIEW__)
          auto Jx_loc = scatter_Jx_loc.access();
          auto Jy_loc = scatter_Jy_loc.access();
          auto Jz_loc = scatter_Jz_loc.access();
#endif

          const double charge_weight = inv_cell_volume_x_q * w(part);

          // gamma-factor
          const double gamma_inv =
            1 / Kokkos::sqrt(1 + (mx(part) * mx(part) + my(part) * my(part) + mz(part) * mz(part)));

          const double vx = mx(part) * gamma_inv;
          const double vy = my(part) * gamma_inv;
          const double vz = mz(part) * gamma_inv;

          // Current from the particle
          const double Jxp = vx * charge_weight;
          const double Jyp = vy * charge_weight;
          const double Jzp = vz * charge_weight;

          // Calculate normalized position relative to the patch
          // ixn = (particles_m[is].x(part) ) * params.inv_dx;
          // iyn = (particles_m[is].y(part) ) * params.inv_dy;
          // izn = (particles_m[is].z(part) ) * params.inv_dz;
          const double posxn = (x(part) - 0.5 * dt * vx - xmin) * inv_dx + 1;
          const double posyn = (y(part) - 0.5 * dt * vy - ymin) * inv_dy + 1;
          const double poszn = (z(part) - 0.5 * dt * vz - zmin) * inv_dz + 1;

          // Compute indexes in primal grid
          const int ixp = (int)(Kokkos::floor(posxn));
          const int iyp = (int)(Kokkos::floor(posyn));
          const int izp = (int)(Kokkos::floor(poszn));

          // Compute indexes in dual grid
          // For the current, the dual grid is 0.5 * dx shorter on each side of the grid (if dual
          // directions only)
          const int ixd = (int)(Kokkos::floor(posxn - 0.5));
          const int iyd = (int)(Kokkos::floor(posyn - 0.5));
          const int izd = (int)(Kokkos::floor(poszn - 0.5));

          // Projection particle on currant field
          // Compute interpolation coeff, p = primal, d = dual

          double coeffs[3];

          coeffs[0] = posxn - 0.5 - ixd;
          coeffs[1] = posyn - iyp;
          coeffs[2] = poszn - izp;

          Jx_loc(ixd, iyp, izp) += (1 - coeffs[0]) * (1 - coeffs[1]) * (1 - coeffs[2]) * Jxp;
          Jx_loc(ixd, iyp, izp + 1) += (1 - coeffs[0]) * (1 - coeffs[1]) * (coeffs[2]) * Jxp;
          Jx_loc(ixd, iyp + 1, izp) += (1 - coeffs[0]) * (coeffs[1]) * (1 - coeffs[2]) * Jxp;
          Jx_loc(ixd, iyp + 1, izp + 1) += (1 - coeffs[0]) * (coeffs[1]) * (coeffs[2]) * Jxp;
          Jx_loc(ixd + 1, iyp, izp) += (coeffs[0]) * (1 - coeffs[1]) * (1 - coeffs[2]) * Jxp;
          Jx_loc(ixd + 1, iyp, izp + 1) += (coeffs[0]) * (1 - coeffs[1]) * (coeffs[2]) * Jxp;
          Jx_loc(ixd + 1, iyp + 1, izp) += (coeffs[0]) * (coeffs[1]) * (1 - coeffs[2]) * Jxp;
          Jx_loc(ixd + 1, iyp + 1, izp + 1) += (coeffs[0]) * (coeffs[1]) * (coeffs[2]) * Jxp;

          coeffs[0] = posxn - ixp;
          coeffs[1] = posyn - 0.5 - iyd;
          coeffs[2] = poszn - izp;

          Jy_loc(ixp, iyd, izp) += (1 - coeffs[0]) * (1 - coeffs[1]) * (1 - coeffs[2]) * Jyp;
          Jy_loc(ixp, iyd, izp + 1) += (1 - coeffs[0]) * (1 - coeffs[1]) * (coeffs[2]) * Jyp;
          Jy_loc(ixp, iyd + 1, izp) += (1 - coeffs[0]) * (coeffs[1]) * (1 - coeffs[2]) * Jyp;
          Jy_loc(ixp, iyd + 1, izp + 1) += (1 - coeffs[0]) * (coeffs[1]) * (coeffs[2]) * Jyp;
          Jy_loc(ixp + 1, iyd, izp) += (coeffs[0]) * (1 - coeffs[1]) * (1 - coeffs[2]) * Jyp;
          Jy_loc(ixp + 1, iyd, izp + 1) += (coeffs[0]) * (1 - coeffs[1]) * (coeffs[2]) * Jyp;
          Jy_loc(ixp + 1, iyd + 1, izp) += (coeffs[0]) * (coeffs[1]) * (1 - coeffs[2]) * Jyp;
          Jy_loc(ixp + 1, iyd + 1, izp + 1) += (coeffs[0]) * (coeffs[1]) * (coeffs[2]) * Jyp;

          coeffs[0] = posxn - ixp;
          coeffs[1] = posyn - iyp;
          coeffs[2] = poszn - 0.5 - izd;

          Jz_loc(ixp, iyp, izd) += (1 - coeffs[0]) * (1 - coeffs[1]) * (1 - coeffs[2]) * Jzp;
          Jz_loc(ixp, iyp, izd + 1) += (1 - coeffs[0]) * (1 - coeffs[1]) * (coeffs[2]) * Jzp;
          Jz_loc(ixp, iyp + 1, izd) += (1 - coeffs[0]) * (coeffs[1]) * (1 - coeffs[2]) * Jzp;
          Jz_loc(ixp, iyp + 1, izd + 1) += (1 - coeffs[0]) * (coeffs[1]) * (coeffs[2]) * Jzp;
          Jz_loc(ixp + 1, iyp, izd) += (coeffs[0]) * (1 - coeffs[1]) * (1 - coeffs[2]) * Jzp;
          Jz_loc(ixp + 1, iyp, izd + 1) += (coeffs[0]) * (1 - coeffs[1]) * (coeffs[2]) * Jzp;
          Jz_loc(ixp + 1, iyp + 1, izd) += (coeffs[0]) * (coeffs[1]) * (1 - coeffs[2]) * Jzp;
          Jz_loc(ixp + 1, iyp + 1, izd + 1) += (coeffs[0]) * (coeffs[1]) * (coeffs[2]) * Jzp;
        } // end for each particles

      );

      Kokkos::fence();

#if defined(__MINIPIC_KOKKOS_SCATTERVIEW__)
      Kokkos::Experimental::contribute(Jx_species, scatter_Jx_loc);
      Kokkos::Experimental::contribute(Jy_species, scatter_Jy_loc);
      Kokkos::Experimental::contribute(Jz_species, scatter_Jz_loc);
#endif

      patch.projected_[is] = true;

    } else {

      patch.projected_[is] = false;

    } // end if n_particles > 0
  } // end loop species
}

// _______________________________________________________________________
//
//! \brief Current projection directly in the global array
//! \param[in] params constant global parameters
//! \param[in] em electromagnetic fields
//! \param[in] patch current patch to handle
// _______________________________________________________________________
void project(Params &params, ElectroMagn &em, Patch &patch) {

  device_field_t Jx_device = em.Jx_m.data_m;
  device_field_t Jy_device = em.Jx_m.data_m;
  device_field_t Jz_device = em.Jx_m.data_m;

#if defined(__MINIPIC_KOKKOS_SCATTERVIEW__)
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

#if (__MINIPIC_DEBUG__)
  int nx_Jx = em.Jx_m.nx_m;
  int ny_Jx = em.Jx_m.ny_m;
  int nz_Jx = em.Jx_m.nz_m;

  int nx_Jy = em.Jy_m.nx_m;
  int ny_Jy = em.Jy_m.ny_m;
  int nz_Jy = em.Jy_m.nz_m;

#endif

  for (int is = 0; is < patch.n_species_m; is++) {

    const int n_particles            = patch.particles_m[is].size();
    const double inv_cell_volume_x_q = params.inv_cell_volume * patch.particles_m[is].charge_m;
    // double m       = particles_m[is].mass_m;

    device_vector_t w = patch.particles_m[is].weight_.data_;

    device_vector_t x = patch.particles_m[is].x_.data_;
    device_vector_t y = patch.particles_m[is].y_.data_;
    device_vector_t z = patch.particles_m[is].z_.data_;

    device_vector_t mx = patch.particles_m[is].mx_.data_;
    device_vector_t my = patch.particles_m[is].my_.data_;
    device_vector_t mz = patch.particles_m[is].mz_.data_;

    Kokkos::parallel_for(
      n_particles,
      KOKKOS_LAMBDA(const int part) {

#if defined(__MINIPIC_KOKKOS_SCATTERVIEW__)
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
          1 / Kokkos::sqrt(1 + (mx(part) * mx(part) + my(part) * my(part) + mz(part) * mz(part)));

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
        const int ixp = (int)(Kokkos::floor(posxn)); //- i_patch_topology_m * nx_cells_m;
        const int iyp = (int)(Kokkos::floor(posyn)); //- j_patch_topology_m * ny_cells_m;
        const int izp = (int)(Kokkos::floor(poszn)); //- k_patch_topology_m * nz_cells_m;

        // Compute indexes in dual grid
        const int ixd = (int)Kokkos::floor(posxn - 0.5); //- i_patch_topology_m * nx_cells_m;
        const int iyd = (int)Kokkos::floor(posyn - 0.5); //- j_patch_topology_m * ny_cells_m;
        const int izd = (int)Kokkos::floor(poszn - 0.5); //- k_patch_topology_m * nz_cells_m;

#if (__MINIPIC_DEBUG__)
        // Check if the indexes are in the correct range
        if (ixd < 0 || ixd + 1 >= nx_Jx || iyp < 0 || iyp + 1 >= ny_Jx || izp < 0 ||
            izp + 1 >= nz_Jx) {
          Kokkos::printf("Error: part = %d\n", part);
          Kokkos::printf("Error: ixp = %d, iyp = %d, izp = %d\n", ixp, iyp, izp);
          Kokkos::printf("Error: posxn = %f, posyn = %f, poszn = %f\n", posxn, posyn, poszn);
          Kokkos::printf("Error: x = %f, y = %f, z = %f\n", x(part), y(part), z(part));
          Kokkos::printf("Error: vx = %f, vy = %f, vz = %f\n", vx, vy, vz);
          Kokkos::printf("Error: gamma_inv = %f\n", gamma_inv);
        }
#endif

        // Projection particle on currant field
        // Compute interpolation coeff, p = primal, d = dual

        double coeffs[3];

        coeffs[0] = posxn - 0.5 - ixd;
        coeffs[1] = posyn - iyp;
        coeffs[2] = poszn - izp;

        // #if defined(__MINIPIC_KOKKOS_SCATTERVIEW__)
        Jx(ixd, iyp, izp) += (1 - coeffs[0]) * (1 - coeffs[1]) * (1 - coeffs[2]) * Jxp;
        Jx(ixd, iyp, izp + 1) += (1 - coeffs[0]) * (1 - coeffs[1]) * (coeffs[2]) * Jxp;
        Jx(ixd, iyp + 1, izp) += (1 - coeffs[0]) * (coeffs[1]) * (1 - coeffs[2]) * Jxp;
        Jx(ixd, iyp + 1, izp + 1) += (1 - coeffs[0]) * (coeffs[1]) * (coeffs[2]) * Jxp;
        Jx(ixd + 1, iyp, izp) += (coeffs[0]) * (1 - coeffs[1]) * (1 - coeffs[2]) * Jxp;
        Jx(ixd + 1, iyp, izp + 1) += (coeffs[0]) * (1 - coeffs[1]) * (coeffs[2]) * Jxp;
        Jx(ixd + 1, iyp + 1, izp) += (coeffs[0]) * (coeffs[1]) * (1 - coeffs[2]) * Jxp;
        Jx(ixd + 1, iyp + 1, izp + 1) += (coeffs[0]) * (coeffs[1]) * (coeffs[2]) * Jxp;
        // #else
        //  // Use kokkos atomics
        //  Kokkos::atomic_add(&Jx_device(ixd, iyp, izp), (1 - coeffs[0]) * (1 - coeffs[1]) * (1 -
        //  coeffs[2]) * Jxp); Kokkos::atomic_add(&Jx_device(ixd, iyp, izp + 1), (1 - coeffs[0]) *
        //  (1 - coeffs[1]) * (coeffs[2]) * Jxp); Kokkos::atomic_add(&Jx_device(ixd, iyp + 1, izp),
        //  (1 - coeffs[0]) * (coeffs[1]) * (1 - coeffs[2]) * Jxp);
        //  Kokkos::atomic_add(&Jx_device(ixd, iyp + 1, izp + 1), (1 - coeffs[0]) * (coeffs[1]) *
        //  (coeffs[2]) * Jxp); Kokkos::atomic_add(&Jx_device(ixd + 1, iyp, izp), (coeffs[0]) * (1 -
        //  coeffs[1]) * (1 - coeffs[2]) * Jxp); Kokkos::atomic_add(&Jx_device(ixd + 1, iyp, izp +
        //  1), (coeffs[0]) * (1 - coeffs[1]) * (coeffs[2]) * Jxp);
        //  Kokkos::atomic_add(&Jx_device(ixd + 1, iyp + 1, izp), (coeffs[0]) * (coeffs[1]) * (1 -
        //  coeffs[2]) * Jxp); Kokkos::atomic_add(&Jx_device(ixd + 1, iyp + 1, izp + 1), (coeffs[0])
        //  * (coeffs[1]) * (coeffs[2]) * Jxp);
        // #endif

        coeffs[0] = posxn - ixp;
        coeffs[1] = posyn - 0.5 - iyd;
        coeffs[2] = poszn - izp;

        // #if defined(__MINIPIC_KOKKOS_SCATTERVIEW__)
        Jy(ixp, iyd, izp) += (1 - coeffs[0]) * (1 - coeffs[1]) * (1 - coeffs[2]) * Jyp;
        Jy(ixp, iyd, izp + 1) += (1 - coeffs[0]) * (1 - coeffs[1]) * (coeffs[2]) * Jyp;
        Jy(ixp, iyd + 1, izp) += (1 - coeffs[0]) * (coeffs[1]) * (1 - coeffs[2]) * Jyp;
        Jy(ixp, iyd + 1, izp + 1) += (1 - coeffs[0]) * (coeffs[1]) * (coeffs[2]) * Jyp;
        Jy(ixp + 1, iyd, izp) += (coeffs[0]) * (1 - coeffs[1]) * (1 - coeffs[2]) * Jyp;
        Jy(ixp + 1, iyd, izp + 1) += (coeffs[0]) * (1 - coeffs[1]) * (coeffs[2]) * Jyp;
        Jy(ixp + 1, iyd + 1, izp) += (coeffs[0]) * (coeffs[1]) * (1 - coeffs[2]) * Jyp;
        Jy(ixp + 1, iyd + 1, izp + 1) += (coeffs[0]) * (coeffs[1]) * (coeffs[2]) * Jyp;
        // #else
        //  Use kokkos atomics
        //  Kokkos::atomic_add(&Jy_device(ixp, iyd, izp), (1 - coeffs[0]) * (1 - coeffs[1]) * (1 -
        //  coeffs[2]) * Jyp); Kokkos::atomic_add(&Jy_device(ixp, iyd, izp + 1), (1 - coeffs[0]) *
        //  (1 - coeffs[1]) * (coeffs[2]) * Jyp); Kokkos::atomic_add(&Jy_device(ixp, iyd + 1, izp),
        //  (1 - coeffs[0]) * (coeffs[1]) * (1 - coeffs[2]) * Jyp);
        //  Kokkos::atomic_add(&Jy_device(ixp, iyd + 1, izp + 1), (1 - coeffs[0]) * (coeffs[1]) *
        //  (coeffs[2]) * Jyp); Kokkos::atomic_add(&Jy_device(ixp + 1, iyd, izp), (coeffs[0]) * (1 -
        //  coeffs[1]) * (1 - coeffs[2]) * Jyp); Kokkos::atomic_add(&Jy_device(ixp + 1, iyd, izp +
        //  1), (coeffs[0]) * (1 - coeffs[1]) * (coeffs[2]) * Jyp);
        //  Kokkos::atomic_add(&Jy_device(ixp + 1, iyd + 1, izp), (coeffs[0]) * (coeffs[1]) * (1 -
        //  coeffs[2]) * Jyp); Kokkos::atomic_add(&Jy_device(ixp + 1, iyd + 1, izp + 1), (coeffs[0])
        //  * (coeffs[1]) * (coeffs[2]) * Jyp);
        // #endif

        coeffs[0] = posxn - ixp;
        coeffs[1] = posyn - iyp;
        coeffs[2] = poszn - 0.5 - izd;

        // #if defined(__MINIPIC_KOKKOS_SCATTERVIEW__)
        Jz(ixp, iyp, izd) += (1 - coeffs[0]) * (1 - coeffs[1]) * (1 - coeffs[2]) * Jzp;
        Jz(ixp, iyp, izd + 1) += (1 - coeffs[0]) * (1 - coeffs[1]) * (coeffs[2]) * Jzp;
        Jz(ixp, iyp + 1, izd) += (1 - coeffs[0]) * (coeffs[1]) * (1 - coeffs[2]) * Jzp;
        Jz(ixp, iyp + 1, izd + 1) += (1 - coeffs[0]) * (coeffs[1]) * (coeffs[2]) * Jzp;
        Jz(ixp + 1, iyp, izd) += (coeffs[0]) * (1 - coeffs[1]) * (1 - coeffs[2]) * Jzp;
        Jz(ixp + 1, iyp, izd + 1) += (coeffs[0]) * (1 - coeffs[1]) * (coeffs[2]) * Jzp;
        Jz(ixp + 1, iyp + 1, izd) += (coeffs[0]) * (coeffs[1]) * (1 - coeffs[2]) * Jzp;
        Jz(ixp + 1, iyp + 1, izd + 1) += (coeffs[0]) * (coeffs[1]) * (coeffs[2]) * Jzp;
        // #else
        //  // Use kokkos atomics
        //  Kokkos::atomic_add(&Jz_device(ixp, iyp, izd), (1 - coeffs[0]) * (1 - coeffs[1]) * (1 -
        //  coeffs[2]) * Jzp); Kokkos::atomic_add(&Jz_device(ixp, iyp, izd + 1), (1 - coeffs[0]) *
        //  (1 - coeffs[1]) * (coeffs[2]) * Jzp); Kokkos::atomic_add(&Jz_device(ixp, iyp + 1, izd),
        //  (1 - coeffs[0]) * (coeffs[1]) * (1 - coeffs[2]) * Jzp);
        //  Kokkos::atomic_add(&Jz_device(ixp, iyp + 1, izd + 1), (1 - coeffs[0]) * (coeffs[1]) *
        //  (coeffs[2]) * Jzp); Kokkos::atomic_add(&Jz_device(ixp + 1, iyp, izd), (coeffs[0]) * (1 -
        //  coeffs[1]) * (1 - coeffs[2]) * Jzp); Kokkos::atomic_add(&Jz_device(ixp + 1, iyp, izd +
        //  1), (coeffs[0]) * (1 - coeffs[1]) * (coeffs[2]) * Jzp);
        //  Kokkos::atomic_add(&Jz_device(ixp + 1, iyp + 1, izd), (coeffs[0]) * (coeffs[1]) * (1 -
        //  coeffs[2]) * Jzp); Kokkos::atomic_add(&Jz_device(ixp + 1, iyp + 1, izd + 1), (coeffs[0])
        //  * (coeffs[1]) * (coeffs[2]) * Jzp);
        // #endif
      } // end for each particles
    );

    Kokkos::fence();
#if defined(__MINIPIC_KOKKOS_SCATTERVIEW__)
    Kokkos::Experimental::contribute(Jx_device, scatter_Jx);
    Kokkos::Experimental::contribute(Jy_device, scatter_Jy);
    Kokkos::Experimental::contribute(Jz_device, scatter_Jz);
#endif
    // particles_m[is].sync(minipic::device, minipic::host);
  }
}

// _______________________________________________________
//
//! \brief Solve Maxwell equations to compute EM fields
//! \param params global parameters
// _______________________________________________________
auto solve_maxwell(const Params &params, ElectroMagn &em) -> void {

  const double dt         = params.dt;
  const double dt_over_dx = params.dt * params.inv_dx;
  const double dt_over_dy = params.dt * params.inv_dy;
  const double dt_over_dz = params.dt * params.inv_dz;

  /////     Solve Maxwell Ampere (E)
  // Electric field Ex (d,p,p)

  device_field_t Jx = em.Jx_m.data_m;
  device_field_t Jy = em.Jy_m.data_m;
  device_field_t Jz = em.Jz_m.data_m;

  device_field_t Ex = em.Ex_m.data_m;
  device_field_t Ey = em.Ey_m.data_m;
  device_field_t Ez = em.Ez_m.data_m;

  device_field_t Bx = em.Bx_m.data_m;
  device_field_t By = em.By_m.data_m;
  device_field_t Bz = em.Bz_m.data_m;

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

// _______________________________________________________________
//
//! \brief Boundaries condition on the global grid
//! \param[in] Params & params - global constant parameters
// _______________________________________________________________
void currentBC(Params &params, ElectroMagn &em) {

  if (params.boundary_condition == "periodic") {

    device_field_t Jx = em.Jx_m.data_m;
    device_field_t Jy = em.Jy_m.data_m;
    device_field_t Jz = em.Jz_m.data_m;

    const auto nx_Jx = em.Jx_m.nx();
    const auto ny_Jx = em.Jx_m.ny();
    const auto nz_Jx = em.Jx_m.nz();

    const auto nx_Jy = em.Jy_m.nx();
    const auto ny_Jy = em.Jy_m.ny();
    const auto nz_Jy = em.Jy_m.nz();

    const auto nx_Jz = em.Jz_m.nx();
    const auto ny_Jz = em.Jz_m.ny();
    const auto nz_Jz = em.Jz_m.nz();

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
      mdrange_policy({0, 0}, {nx_Jx, nx_Jx}),
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

// _______________________________________________________________
//
//! \brief Boundaries condition on the global grid
//! \param[in] Params & params - global constant parameters
// _______________________________________________________________
auto solveBC(Params &params, ElectroMagn &em) -> void {
  typedef Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<2>> mdrange_policy;

  if (params.boundary_condition == "periodic") {

    device_field_t Bx = em.Bx_m.data_m;
    device_field_t By = em.By_m.data_m;
    device_field_t Bz = em.Bz_m.data_m;

    const auto nx_Bx = em.Bx_m.nx();
    const auto ny_Bx = em.Bx_m.ny();
    const auto nz_Bx = em.Bx_m.nz();

    const auto nx_By = em.By_m.nx();
    const auto ny_By = em.By_m.ny();
    const auto nz_By = em.By_m.nz();

    const auto nx_Bz = em.Bz_m.nx();
    const auto ny_Bz = em.Bz_m.ny();
    const auto nz_Bz = em.Bz_m.nz();

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

    device_field_t Bx = em.Bx_m.data_m;
    device_field_t By = em.By_m.data_m;
    device_field_t Bz = em.Bz_m.data_m;

    const auto nx_Bx = em.Bx_m.nx();
    const auto ny_Bx = em.Bx_m.ny();
    const auto nz_Bx = em.Bx_m.nz();

    const auto nx_By = em.By_m.nx();
    const auto ny_By = em.By_m.ny();
    const auto nz_By = em.By_m.nz();

    const auto nx_Bz = em.Bz_m.nx();
    const auto ny_Bz = em.Bz_m.ny();
    const auto nz_Bz = em.Bz_m.nz();

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

// ______________________________________________________
//
//! \brief Sum all species local current grids in local grid
//! \param[in] patch  current patch to handle
// ______________________________________________________
auto reduc_current(Patch &patch) -> void {

  for (int is = 1; is < patch.n_species_m; is++) {

    // Only if particles projected
    if (patch.projected_[is]) {

      device_field_t Jx_0  = patch.vec_Jx_m[0].data_m;
      device_field_t Jx_is = patch.vec_Jx_m[is].data_m;

      device_field_t Jy_0  = patch.vec_Jy_m[0].data_m;
      device_field_t Jy_is = patch.vec_Jy_m[is].data_m;

      device_field_t Jz_0  = patch.vec_Jz_m[0].data_m;
      device_field_t Jz_is = patch.vec_Jz_m[is].data_m;

      typedef Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<3>> mdrange_policy;
      Kokkos::parallel_for(
        mdrange_policy({0, 0, 0},
                       {patch.vec_Jx_m[is].nx(), patch.vec_Jx_m[is].ny(), patch.vec_Jx_m[is].nz()}),
        KOKKOS_LAMBDA(const int ix, const int iy, const int iz) {
          Jx_0(ix, iy, iz) += Jx_is(ix, iy, iz);
        });

      typedef Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<3>> mdrange_policy;
      Kokkos::parallel_for(
        mdrange_policy({0, 0, 0},
                       {patch.vec_Jy_m[is].nx(), patch.vec_Jy_m[is].ny(), patch.vec_Jy_m[is].nz()}),
        KOKKOS_LAMBDA(const int ix, const int iy, const int iz) {
          Jy_0(ix, iy, iz) += Jy_is(ix, iy, iz);
        });

      typedef Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<3>> mdrange_policy;
      Kokkos::parallel_for(
        mdrange_policy({0, 0, 0},
                       {patch.vec_Jz_m[is].nx(), patch.vec_Jz_m[is].ny(), patch.vec_Jz_m[is].nz()}),
        KOKKOS_LAMBDA(const int ix, const int iy, const int iz) {
          Jz_0(ix, iy, iz) += Jz_is(ix, iy, iz);
        });
      Kokkos::fence();

    } // end check if particles
  } // end for species
}

// ____________________________________________________________________________
//! \brief Copy all local current grid in the global grid
//! \param[in] ElectroMagn & em - global electromagnetic fields
//! \param[in] Patch & patch - current patch
// ____________________________________________________________________________
auto local2global(ElectroMagn &em, Patch &patch) -> void {

  bool projected = false;

  for (int is = 0; is < patch.n_species_m; is++) {
    projected = projected || patch.projected_[is];
  }

  // projection only if particles in this patch
  if (projected) {

    // for (int is = 0; is < n_species_m; is++) {
    const int i_global_p = patch.ix_origin_m;
    const int j_global_p = patch.iy_origin_m;
    const int k_global_p = patch.iz_origin_m;
    const int i_global_d = patch.ix_origin_m;
    const int j_global_d = patch.iy_origin_m;
    const int k_global_d = patch.iz_origin_m;

    // std::cerr << i_global_p  << " " << j_global_p << " " << k_global_p << std::endl;
    // std::cerr << "nx: " << vec_Jx_m[0].nx()  << " " << em.Jx_m.nx() << " " << std::endl;

    device_field_t Jx_0 = patch.vec_Jx_m[0].data_m;
    device_field_t Jx   = em.Jx_m.data_m;

    device_field_t Jy_0 = patch.vec_Jy_m[0].data_m;
    device_field_t Jy   = em.Jy_m.data_m;

    device_field_t Jz_0 = patch.vec_Jz_m[0].data_m;
    device_field_t Jz   = em.Jz_m.data_m;

    typedef Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<3>> mdrange_policy;
    Kokkos::parallel_for(
      mdrange_policy({0, 0, 0},
                     {patch.vec_Jx_m[0].nx(), patch.vec_Jx_m[0].ny(), patch.vec_Jx_m[0].nz()}),
      KOKKOS_LAMBDA(const int ix, const int iy, const int iz) {
        Jx(i_global_d + ix, j_global_p + iy, k_global_p + iz) += Jx_0(ix, iy, iz);
      });

    typedef Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<3>> mdrange_policy;
    Kokkos::parallel_for(
      mdrange_policy({0, 0, 0},
                     {patch.vec_Jy_m[0].nx(), patch.vec_Jy_m[0].ny(), patch.vec_Jy_m[0].nz()}),
      KOKKOS_LAMBDA(const int ix, const int iy, const int iz) {
        Jy(i_global_p + ix, j_global_d + iy, k_global_p + iz) += Jy_0(ix, iy, iz);
      });

    typedef Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<3>> mdrange_policy;
    Kokkos::parallel_for(
      mdrange_policy({0, 0, 0},
                     {patch.vec_Jz_m[0].nx(), patch.vec_Jz_m[0].ny(), patch.vec_Jz_m[0].nz()}),
      KOKKOS_LAMBDA(const int ix, const int iy, const int iz) {
        Jz(i_global_p + ix, j_global_p + iy, k_global_d + iz) += Jz_0(ix, iy, iz);
      });
    Kokkos::fence();

  } // end if total_particles
}

// ____________________________________________________________________________
//! \brief Emit a laser field in the x direction using an antenna
//! \param[in] Params & params - global constant parameters
//! \param[in] profile - (std::function<double(double y, double z, double t)>) profile of the
//! antenna \param[in] x - (double) position of the antenna \param[in] double t - (double) current
//! time
// ____________________________________________________________________________
auto antenna(Params &params,
             ElectroMagn &em,
             std::function<double(double, double, double)> profile,
             double x,
             double t) -> void {

  em.Jz_m.sync(minipic::device, minipic::host);

  Field<mini_float> *J = &em.Jz_m;

  const int ix = Kokkos::floor((x - params.inf_x - J->dual_x_m * 0.5 * params.dx) / params.dx);

  const double yfs = 0.5 * params.Ly + params.inf_y;
  const double zfs = 0.5 * params.Lz + params.inf_z;

  for (int iy = 0; iy < J->ny_m; ++iy) {
    for (int iz = 0; iz < J->nz_m; ++iz) {

      const double y = (iy - J->dual_y_m * 0.5) * params.dy + params.inf_y - yfs;
      const double z = (iz - J->dual_z_m * 0.5) * params.dz + params.inf_z - zfs;

      (*J)(ix, iy, iz) = profile(y, z, t);
    }
  }

  em.Jz_m.sync(minipic::host, minipic::device);

} // end antenna

} // end namespace operators

#endif // OPERATORS_H

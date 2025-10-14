
/* _____________________________________________________________________ */
//! \file SubDomain.hpp

//! \brief Management of the full domain

/* _____________________________________________________________________ */

#pragma once

#include <vector>

#include "Diagnotics.hpp"
#include "Operators.hpp"
#include "Params.hpp"
#include "Patch.hpp"
#include "Timers.hpp"

//! \brief Wrapper class to clean main
class SubDomain {
public:
  //! List of Patch in this subdomain, linearized by Z, Z fast
  std::vector<Patch> patches_;

  // Init global fields
  ElectroMagn em_;

  // ______________________________________________________
  //
  //! \brief Alloc memory to store all the patch
  //! \param[in] params global parameters
  // ______________________________________________________
  void allocate(Params &params) {

    std::cout << params.seperator(50) << std::endl << std::endl;
    std::cout << " Initialization" << std::endl << std::endl;

    // Allocate global fields
    em_.allocate(params);

    const double memory_consumption =
      (em_.Ex_m.size() + em_.Ey_m.size() + em_.Ez_m.size() + em_.Bx_m.size() + em_.By_m.size() +
       em_.Bz_m.size() +
       (em_.Jx_m.size() + em_.Jy_m.size() + em_.Jz_m.size()) * (params.species_names_.size() + 1)) *
      8. / (1024. * 1024);

    std::cout << " Field grids: " << memory_consumption << " Mb" << std::endl << std::endl;

    // Allocate and initialize particles for each patch on host
    patches_.resize(params.N_patches);
    for (int i = 0; i < params.nx_patch; i++) {
      for (int j = 0; j < params.ny_patch; j++) {
        for (int k = 0; k < params.nz_patch; k++) {
          int idx = i * params.nz_patch * params.ny_patch + j * params.nz_patch + k;
          // Memory allocate for all particles and local fields
          patches_[idx].allocate(params, i, j, k);

          // Particles position and momentum initialization
          patches_[idx].initialize_particles(params);
        }
      }
    }

    // Momentum correction (to respect the leap frog scheme)
    if (params.momentum_correction) {

      std::cout << " > Apply momentum correction "
                << "\n"
                << std::endl;

      for (size_t ip = 0; ip < patches_.size(); ip++) {
        operators::interpolate(em_, patches_[ip]);
        operators::push_momentum(patches_[ip], -0.5 * params.dt);
      }
    }

    // For each species, print :
    // - total number of particles
    for (size_t is = 0; is < params.species_names_.size(); ++is) {
      unsigned int total_number_of_particles = 0;
      double total_particle_energy           = 0;
      for (size_t idx_patch = 0; idx_patch < patches_.size(); idx_patch++) {
        total_number_of_particles += patches_[idx_patch].particles_m[is].size();
        total_particle_energy +=
          patches_[idx_patch].particles_m[is].get_kinetic_energy(minipic::host);
      }
      std::cout << " Species " << params.species_names_[is] << std::endl;

      const double memory_consumption = total_number_of_particles * 14. * 8. / (1024. * 1024);

      std::cout << " - Initialized particles: " << total_number_of_particles << std::endl;
      std::cout << " - Total kinetic energy: " << total_particle_energy << std::endl;
      std::cout << " - Memory footprint: " << memory_consumption << " Mb" << std::endl;
    }

    // Checksum for field

    auto sum_Ex_on_host = em_.Ex_m.sum(1, minipic::host);
    auto sum_Ey_on_host = em_.Ey_m.sum(1, minipic::host);
    auto sum_Ez_on_host = em_.Ez_m.sum(1, minipic::host);

    auto sum_Bx_on_host = em_.Bx_m.sum(1, minipic::host);
    auto sum_By_on_host = em_.By_m.sum(1, minipic::host);
    auto sum_Bz_on_host = em_.Bz_m.sum(1, minipic::host);

    auto sum_Ex_on_device = em_.Ex_m.sum(1, minipic::device);
    auto sum_Ey_on_device = em_.Ey_m.sum(1, minipic::device);
    auto sum_Ez_on_device = em_.Ez_m.sum(1, minipic::device);

    auto sum_Bx_on_device = em_.Bx_m.sum(1, minipic::device);
    auto sum_By_on_device = em_.By_m.sum(1, minipic::device);
    auto sum_Bz_on_device = em_.Bz_m.sum(1, minipic::device);

    static const int p = 3;

    std::cout << std::endl;
    std::cout << " -------------------------------- |" << std::endl;
    std::cout << " Check sum for fields             |" << std::endl;
    std::cout << " -------------------------------- |" << std::endl;
    std::cout << " Field  | Host       | Device     |" << std::endl;
    std::cout << " -------------------------------- |" << std::endl;
    std::cout << " Ex     | " << std::setw(10) << std::scientific << std::setprecision(p)
              << sum_Ex_on_host << " | " << std::setw(10) << std::scientific << std::setprecision(p)
              << sum_Ex_on_device << " | " << std::endl;
    std::cout << " Ey     | " << std::setw(10) << std::scientific << std::setprecision(p)
              << sum_Ey_on_host << " | " << std::setw(10) << std::scientific << std::setprecision(p)
              << sum_Ey_on_device << " | " << std::endl;
    std::cout << " Ez     | " << std::setw(10) << std::scientific << std::setprecision(p)
              << sum_Ez_on_host << " | " << std::setw(10) << std::scientific << std::setprecision(p)
              << sum_Ez_on_device << " | " << std::endl;
    std::cout << " Bx     | " << std::setw(10) << std::scientific << std::setprecision(p)
              << sum_Bx_on_host << " | " << std::setw(10) << std::scientific << std::setprecision(p)
              << sum_Bx_on_device << " | " << std::endl;
    std::cout << " By     | " << std::setw(10) << std::scientific << std::setprecision(p)
              << sum_By_on_host << " | " << std::setw(10) << std::scientific << std::setprecision(p)
              << sum_By_on_device << " | " << std::endl;
    std::cout << " Bz     | " << std::setw(10) << std::scientific << std::setprecision(p)
              << sum_Bz_on_host << " | " << std::setw(10) << std::scientific << std::setprecision(p)
              << sum_Bz_on_device << " | " << std::endl;

    // Checksum for particles

    double sum_device[13] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    double sum_host[13]   = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    static const std::string vector_name[13] =
      {"weight", "x", "y", "z", "mx", "my", "mz", "Ex", "Ey", "Ez", "Bx", "By", "Bz"};
    for (size_t ip = 0; ip < patches_.size(); ip++) {
      for (size_t is = 0; is < params.species_names_.size(); ++is) {

        sum_host[0] += patches_[ip].particles_m[is].weight_.sum(1, minipic::host);
        sum_host[1] += patches_[ip].particles_m[is].x_.sum(1, minipic::host);
        sum_host[2] += patches_[ip].particles_m[is].y_.sum(1, minipic::host);
        sum_host[3] += patches_[ip].particles_m[is].z_.sum(1, minipic::host);
        sum_host[4] += patches_[ip].particles_m[is].mx_.sum(1, minipic::host);
        sum_host[5] += patches_[ip].particles_m[is].my_.sum(1, minipic::host);
        sum_host[6] += patches_[ip].particles_m[is].mz_.sum(1, minipic::host);
        sum_host[7] += patches_[ip].particles_m[is].Ex_.sum(1, minipic::host);
        sum_host[8] += patches_[ip].particles_m[is].Ey_.sum(1, minipic::host);
        sum_host[9] += patches_[ip].particles_m[is].Ez_.sum(1, minipic::host);
        sum_host[10] += patches_[ip].particles_m[is].Bx_.sum(1, minipic::host);
        sum_host[11] += patches_[ip].particles_m[is].By_.sum(1, minipic::host);
        sum_host[12] += patches_[ip].particles_m[is].Bz_.sum(1, minipic::host);

        sum_device[0] += patches_[ip].particles_m[is].weight_.sum(1, minipic::device);
        sum_device[1] += patches_[ip].particles_m[is].x_.sum(1, minipic::device);
        sum_device[2] += patches_[ip].particles_m[is].y_.sum(1, minipic::device);
        sum_device[3] += patches_[ip].particles_m[is].z_.sum(1, minipic::device);
        sum_device[4] += patches_[ip].particles_m[is].mx_.sum(1, minipic::device);
        sum_device[5] += patches_[ip].particles_m[is].my_.sum(1, minipic::device);
        sum_device[6] += patches_[ip].particles_m[is].mz_.sum(1, minipic::device);
        sum_device[7] += patches_[ip].particles_m[is].Ex_.sum(1, minipic::device);
        sum_device[8] += patches_[ip].particles_m[is].Ey_.sum(1, minipic::device);
        sum_device[9] += patches_[ip].particles_m[is].Ez_.sum(1, minipic::device);
        sum_device[10] += patches_[ip].particles_m[is].Bx_.sum(1, minipic::device);
        sum_device[11] += patches_[ip].particles_m[is].By_.sum(1, minipic::device);
        sum_device[12] += patches_[ip].particles_m[is].Bz_.sum(1, minipic::device);
      }
    }

    std::cout << std::endl;
    std::cout << " -------------------------------- |" << std::endl;
    std::cout << " Check sum for particles          |" << std::endl;
    std::cout << " -------------------------------- |" << std::endl;
    std::cout << " vector | Host       | Device     |" << std::endl;
    std::cout << " -------------------------------- |" << std::endl;

    for (int i = 0; i < 13; i++) {
      std::cout << " " << std::setw(6) << vector_name[i] << " | " << std::setw(10)
                << std::scientific << std::setprecision(p) << sum_host[i] << " | " << std::setw(10)
                << std::scientific << std::setprecision(p) << sum_device[i] << " | " << std::endl;
    }
  }

  // ______________________________________________________________________________
  //
  //! \brief Perform a single PIC iteration
  //! \param[in] Params&  global parameters
  //! \param[in] int it iteration number
  // ______________________________________________________________________________
  void iterate(Params &params, int it) {

    if (params.current_projection || params.n_particles > 0) {

      DEBUG("  -> start reset current");

      em_.reset_currents(minipic::device);

      DEBUG("  -> stop reset current");
    }

    for (size_t idx_patch = 0; idx_patch < patches_.size(); idx_patch++) {

      // Interpolate from global field to particles in patch
      DEBUG("  -> start interpolate for patch " << idx_patch);

      operators::interpolate(em_, patches_[idx_patch]);

      DEBUG("  -> stop interpolate");

      // Push all particles in patch
      DEBUG("  -> start push for patch " << idx_patch);

      operators::push(patches_[idx_patch], params.dt);

      DEBUG("  -> stop push");

      // Do boundary conditions on global domain
      DEBUG("  -> Patch " << idx_patch << ": start pushBC");

      operators::pushBC(params, patches_[idx_patch]);

      DEBUG("  -> stop pushBC");


#if defined(__MINIPIC_DEBUG__)
      // check particles
      for (auto is = 0; is < params.species_names_.size(); ++is) {
        patches_[idx_patch].particles_m[is].check(patches_[idx_patch].inf_m[0],
                                                  patches_[idx_patch].sup_m[0],
                                                  patches_[idx_patch].inf_m[1],
                                                  patches_[idx_patch].sup_m[1],
                                                  patches_[idx_patch].inf_m[2],
                                                  patches_[idx_patch].sup_m[2]);
      }
#endif

      // Projection in local field
      if (params.current_projection) {

        // #if defined(__MINIPIC_KOKKOS__)

        // Projection directly in the global grid
        // patches_[idx_patch].project(params, em_);

        // #else

        // Project in buffers local to the patches
        DEBUG("  -> Patch " << idx_patch << ": start project");
        
        operators::project(params, patches_[idx_patch]);

        DEBUG("  -> stop project");

        // #endif
      }

    } // end for patches

    // __________________________________________________________________
    // Sum all species contribution in the local and global current grids

    if (params.current_projection || params.n_particles > 0) {

      for (size_t idx_patch = 0; idx_patch < patches_.size(); idx_patch++) {

        // Projection directly in the global grid
        // subdomain.patches_[idx_patch].project(param, em_);

        // Projection in local field
        // patches_[idx_patch].project(params);

        // Sum all species contribution in the local fields
        DEBUG("  -> Patch " << idx_patch << ": start reduction");

        operators::reduc_current(patches_[idx_patch]);

        DEBUG("  -> Patch " << idx_patch << ": end reduction");

      }

      for (size_t idx_patch = 0; idx_patch < patches_.size(); idx_patch++) {
        
        // Copy all local fields in the global fields
        DEBUG("  -> Patch " << idx_patch << ": start local 2 global");

        operators::local2global(em_, patches_[idx_patch]);

        DEBUG("  -> Patch " << idx_patch << ": end local 2 global");

      }

      // Perform the boundary conditions for current
      DEBUG("  -> start current BC")

      operators::currentBC(params, em_);

      DEBUG("  -> stop current BC")

    } // end if current projection

    // __________________________________________________________________
    // Maxwell solver

    if (params.maxwell_solver) {

      // Generate a laser field with an antenna
      for (size_t iantenna = 0; iantenna < params.antenna_profiles_.size(); iantenna++) {
        operators::antenna(params,
                           em_,
                           params.antenna_profiles_[iantenna],
                           params.antenna_positions_[iantenna],
                           it * params.dt);
      }

      // Solve the Maxwell equation
      DEBUG("  -> start solve Maxwell")

      operators::solve_maxwell(params, em_);

      DEBUG("  -> stop solve Maxwell")

      // Boundary conditions on EM fields
      DEBUG("  -> start solve BC")

      operators::solveBC(params, em_);

      DEBUG("  -> end solve BC")

    } // end test params.maxwell_solver
  }

  // ________________________________________________________________
  //! \brief Perform all diagnostics
  //! \param[in] Params&  global parameters
  //! \param[in] Timers&  timers
  //! \param[in] int it iteration number
  // ________________________________________________________________
  void diagnostics(Params &params, int it) {

    if (params.no_diagnostics_at_init and it == 0) {
      return;
    }

    // __________________________________________________________________
    // Determine species to copy from device to host

    bool *need_species = new bool[params.get_species_number()];
    for (size_t is = 0; is < params.get_species_number(); ++is) {
      need_species[is] = false;
    }

    for (auto particle_binning : params.particle_binning_properties_) {
      if (!(it % particle_binning.period_)) {
        for (auto is : particle_binning.species_indexes_) {
          // if number of particles > 0
          need_species[is] = true;
        }
      }
    }

    if ((params.particle_cloud_period < params.n_it) &&
        (!(it % params.particle_cloud_period) or (it == 0))) {

      for (size_t is = 0; is < params.get_species_number(); ++is) {
        need_species[is] = true;
      }
    }

    for (size_t is = 0; is < params.get_species_number(); ++is) {
      if (need_species[is]) {
        for (size_t ipatch = 0; ipatch < patches_.size(); ++ipatch) {
          patches_[ipatch].particles_m[is].sync(minipic::device, minipic::host);
        }
      }
    }

    delete[] need_species;

    if (!(it % params.field_diagnostics_period)) {
      em_.sync(minipic::device, minipic::host);
    }

    // __________________________________________________________________
    // Start diagnostics

    // Particle binning
    for (auto particle_binning : params.particle_binning_properties_) {

      // for each species index of this diagnostic
      for (auto is : particle_binning.species_indexes_) {

        if (!(it % particle_binning.period_)) {

          // Call the particle binning function using the properties in particle_binning
          Diags::particle_binning(particle_binning.name_,
                                  params,
                                  patches_,
                                  particle_binning.projected_parameter_,
                                  particle_binning.axis_,
                                  particle_binning.n_cells_,
                                  particle_binning.min_,
                                  particle_binning.max_,
                                  is,
                                  it,
                                  particle_binning.format_,
                                  false);

        } // end if test it % period
      }
    } // end loop on particle_binning_properties_

    // Particle Clouds
    if ((params.particle_cloud_period < params.n_it) &&
        (!(it % params.particle_cloud_period) or (it == 0))) {

      for (size_t is = 0; is < params.get_species_number(); ++is) {

        Diags::particle_cloud("cloud", params, patches_, is, it, params.particle_cloud_format);
      }
    }

    // Field diagnostics
    if (!(it % params.field_diagnostics_period)) {

      Diags::fields(params, em_, it, params.field_diagnostics_format);
    }

    // Scalars diagnostics
    if (!(it % params.scalar_diagnostics_period)) {
      for (size_t is = 0; is < params.get_species_number(); ++is) {

        Diags::scalars(params, patches_, is, it);
      }
    }

    if (!(it % params.scalar_diagnostics_period)) {
      {

        Diags::scalars(params, em_, it);
      }
    }

  } // end diagnostics

  // __________________________________________________________________
  //
  //! \brief get the total number of particles
  // __________________________________________________________________
  unsigned int get_total_number_of_particles() {
    unsigned int total_number_of_particles = 0;
    for (size_t idx_patch = 0; idx_patch < patches_.size(); idx_patch++) {
      total_number_of_particles += patches_[idx_patch].get_total_number_of_particles();
    }
    return total_number_of_particles;
  }

}; // end class

/* _____________________________________________________________________ */
//! \file SubDomain.hpp

//! \brief Management of the full domain

/* _____________________________________________________________________ */

#pragma once

#include <vector>

#include "ElectroMagn.hpp"
#include "Particles.hpp"
#include "Diagnostics.hpp"
#include "Params.hpp"
#include "Managers.hpp"

//! \brief Wrapper class to clean main
class SubDomain {
public:

  // Init global fields
  ElectroMagn em_m;

  //! List of species to handle
  std::vector<Particles> particles_m;

  //! Boundaries box of the subdomain
  double inf_m[3];
  double sup_m[3];

  // ______________________________________________________
  //
  //! \brief Alloc memory to store
  //! \param[in] params global parameters
  // ______________________________________________________
  void allocate(Params &params) {

    std::cout << params.seperator(50) << std::endl << std::endl;
    std::cout << " Initialization" << std::endl << std::endl;

    // _____________________________________________________
    // Local parameters

    const int nx_cells_m = params.nx_cells;
    const int ny_cells_m = params.ny_cells;
    const int nz_cells_m = params.nz_cells;

    // Compute boundaries box
    inf_m[0] = params.inf_x ;
    inf_m[1] = params.inf_y ;
    inf_m[2] = params.inf_z ;

    sup_m[0] = inf_m[0] + (params.dx * nx_cells_m);
    sup_m[1] = inf_m[1] + (params.dy * ny_cells_m);
    sup_m[2] = inf_m[2] + (params.dz * nz_cells_m);

    // ______________________________________________________
    // Fields

    // Allocate global fields
    em_m.allocate(params);

    const double memory_consumption =
      (em_m.Ex_m.size() + em_m.Ey_m.size() + em_m.Ez_m.size() + em_m.Bx_m.size() + em_m.By_m.size() +
       em_m.Bz_m.size() +
       (em_m.Jx_m.size() + em_m.Jy_m.size() + em_m.Jz_m.size()) * (params.species_names_m.size() + 1)) *
      8. / (1024. * 1024);

    std::cout << " Field grids: " << memory_consumption << " Mb" << std::endl << std::endl;

    // ______________________________________________________
    // Particles

    unsigned int n_species = params.get_species_number();

    // Alloc vector for each species
    if (n_species > 0) {
      particles_m.resize(n_species);
    }

    for (unsigned int is = 0; is < n_species; is++) {
      int n_particles = params.n_particles_by_species[is] + params.particles_to_add_m.size();

      // Alloc memory to store particles
      particles_m[is].allocate(params.charge_m[is],
                              params.mass_m[is],
                              params.temp_m[is],
                              n_particles,
                              params.inv_cell_volume);
    }

    // Particle initialization

    const int total_cells = params.nx_cells * params.ny_cells * params.nz_cells;

    const double cell_volume = params.cell_volume;

    // buffer to store the number of particles per cells per species
    // Needed for proper init with duplication
    std::vector<int> particles_per_cell_counter(n_species * total_cells);

    for (unsigned int is = 0; is < n_species; is++) {

      double temperature = params.temp_m[is];
      const double mass  = params.mass_m[is];

      // Compute weight
      const int particle_per_cell = params.ppc_m[is];
      const double weight_coef    = cell_volume / particle_per_cell;

      // global particle counter
      unsigned int total_particles_counter = 0;

      // compute the species index for position init
      unsigned int species_index_for_pos_init = 0;
      while (params.species_names_m[species_index_for_pos_init] !=
              params.position_initialization_method_m[is] &&
            (species_index_for_pos_init < is)) {
        ++species_index_for_pos_init;
      }

      // Coefficients for drift velocity
      const double vx = -params.drift_velocity_m[is][0];
      const double vy = -params.drift_velocity_m[is][1];
      const double vz = -params.drift_velocity_m[is][2];

      const double v_drift = vx * vx + vy * vy + vz * vz;

      const double gamma_drift = 1.0 / sqrt(1.0 - v_drift);
      const double gm1         = gamma_drift - 1.0;

      // compute the different component of the Matrix block
      // of the Lorentz transformation (drift velocity correction)
      const double Lxx = 1.0 + gm1 * vx * vx / v_drift;
      const double Lyy = 1.0 + gm1 * vy * vy / v_drift;
      const double Lzz = 1.0 + gm1 * vz * vz / v_drift;
      const double Lxy = gm1 * vx * vy / v_drift;
      const double Lxz = gm1 * vx * vz / v_drift;
      const double Lyz = gm1 * vy * vz / v_drift;

      // If param.position_initialization_method_m[is] is random_per_cell, we init particles randomly
      // with a new seed for each cell.
      if (params.position_initialization_level_m[is] == "cell") {

      // Loop over all cells

        for (std::size_t i = 0; i < params.nx_cells; i++) {
          for (std::size_t j = 0; j < params.ny_cells; j++) {
            for (std::size_t k = 0; k < params.nz_cells; k++) {

              const int i_global = i;
              const int j_global = j;
              const int k_global = k;

              // Local cell index
              const int local_cell_index = i * (params.ny_cells * params.nz_cells) + j * params.nz_cells + k;

              // Global 1d cell index
              const int global_cell_index =
                i_global * (params.ny_cells * params.nz_cells) + j_global * params.nz_cells + k_global;

              Random random(params.seed + global_cell_index + is);

              // counter to compute the number of particles in the current cell
              particles_per_cell_counter[is * total_cells + local_cell_index] = 0;

              // Random Position init
              if (params.position_initialization_method_m[is] == "random") {

                for (auto p = 0; p < particle_per_cell; ++p) {

                  const auto ip = total_particles_counter +
                                  particles_per_cell_counter[is * total_cells + local_cell_index];

                  const double x = (random.draw(0, 1) + i_global) * params.dx;
                  const double y = (random.draw(0, 1) + j_global) * params.dy;
                  const double z = (random.draw(0, 1) + k_global) * params.dz;

                  // Get the density
                  const double w =
                    params.density_profiles_m[is](x / params.Lx, y / params.Ly, z / params.Lz);

                  // only initialize particle if the weight is positive
                  if (w > 1e-10) {

                    particles_m[is].x_h_m(ip) = x;
                    particles_m[is].y_h_m(ip) = y;
                    particles_m[is].z_h_m(ip) = z;

                    particles_m[is].weight_h_m(ip) = w * weight_coef;

                    // increment the number of particles in this cell
                    ++particles_per_cell_counter[is * total_cells + local_cell_index];
                  }

                } // end for particles

                // Init at particle positions of species species_index_for_pos_init
              } else {

                for (auto p = 0;
                    p < particles_per_cell_counter[species_index_for_pos_init * total_cells +
                                                    local_cell_index];
                    ++p) {

                  const auto ip = total_particles_counter + p;

                  // Position
                  particles_m[is].x_h_m(ip) = particles_m[species_index_for_pos_init].x_h_m(ip);
                  particles_m[is].y_h_m(ip) = particles_m[species_index_for_pos_init].y_h_m(ip);
                  particles_m[is].z_h_m(ip) = particles_m[species_index_for_pos_init].z_h_m(ip);

                  // Get the density
                  const double w = params.density_profiles_m[is](
                    static_cast<double>(particles_m[is].x_h_m(ip)) / params.Lx,
                    static_cast<double>(particles_m[is].y_h_m(ip)) / params.Ly,
                    static_cast<double>(particles_m[is].z_h_m(ip)) / params.Lz);

                  // weight
                  particles_m[is].weight_h_m(ip) = w * weight_coef;

                  // increment the number of particles in this cell
                  ++particles_per_cell_counter[is * total_cells + local_cell_index];
                }

              } // end if param.position_initialization_method_m

              for (auto p = 0; p < particles_per_cell_counter[is * total_cells + local_cell_index];
                 ++p) {

                const auto ip = total_particles_counter + p;

                const double energy = Maxwell_Juttner_distribution(temperature / mass, random);

                // Sample angles randomly
                double phi   = std::acos(-random.draw(-1, 1));
                double theta = random.draw(0, 2. * M_PI);
                double psm   = std::sqrt(std::pow(1.0 + energy, 2) - 1.0);

                // Calculate the momentum
                double mx    = psm * cos(theta) * sin(phi);
                double my    = psm * sin(theta) * sin(phi);
                double mz    = psm * cos(phi);
                double gamma = std::sqrt(1.0 + mx * mx + my * my + mz * mz);

                // Add the drift velocity using the Zenitani correction
                // See Zenitani et al. 2015

                if (v_drift > 0) {

                  // Compute the gamma factor using momentum
                  double inverse_gamma = 1. / gamma;

                  const double check_velocity = (vx * mx + vy * my + vz * mz) * inverse_gamma;

                  const double volume_acc = random.draw(0, 1);

                  if (check_velocity > volume_acc) {

                    const double Phi   = std::atan2(sqrt(vx * vx + vy * vy), vz);
                    const double Theta = std::atan2(vy, vx);

                    double vpx = mx * inverse_gamma;
                    double vpy = my * inverse_gamma;
                    double vpz = mz * inverse_gamma;

                    const double vfl =
                      vpx * cos(Theta) * sin(Phi) + vpy * sin(Theta) * sin(Phi) + vpz * cos(Phi);

                    const double vflx = vfl * cos(Theta) * sin(Phi);
                    const double vfly = vfl * sin(Theta) * sin(Phi);
                    const double vflz = vfl * cos(Phi);

                    vpx -= 2. * vflx;
                    vpy -= 2. * vfly;
                    vpz -= 2. * vflz;

                    inverse_gamma = sqrt(1.0 - vpx * vpx - vpy * vpy - vpz * vpz);
                    gamma         = 1. / inverse_gamma;

                    mx = vpx * gamma;
                    my = vpy * gamma;
                    mz = vpz * gamma;

                  } // here ends the corrections by Zenitani

                  particles_m[is].mx_h_m(ip) =
                    -gamma * gamma_drift * vx + Lxx * mx + Lxy * my + Lxz * mz;
                  particles_m[is].my_h_m(ip) =
                    -gamma * gamma_drift * vy + Lxy * my + Lyy * my + Lyz * mz;
                  particles_m[is].mz_h_m(ip) =
                    -gamma * gamma_drift * vz + Lxz * mz + Lyz * my + Lzz * mz;
                } else {
                  particles_m[is].mx_h_m(ip) = mx;
                  particles_m[is].my_h_m(ip) = my;
                  particles_m[is].mz_h_m(ip) = mz;
                }

              } // end for total_particles_counter

              // Add the new particles of this cell to the counter
              total_particles_counter +=
                particles_per_cell_counter[is * total_cells + local_cell_index];

            }
          }
        }

      } // end if position_initialization_level_m == cell

      // Add single particles
      for (size_t ip = 0; ip < params.particles_to_add_m.size(); ++ip) {
        if (params.particles_to_add_m[ip].is_m == is) {

          const double w = params.particles_to_add_m[ip].weight_m;

          const double x = params.particles_to_add_m[ip].x_m;
          const double y = params.particles_to_add_m[ip].y_m;
          const double z = params.particles_to_add_m[ip].z_m;

          const double mx = params.particles_to_add_m[ip].mx_m;
          const double my = params.particles_to_add_m[ip].my_m;
          const double mz = params.particles_to_add_m[ip].mz_m;

          if ((x >= inf_m[0] && x < sup_m[0]) && (y >= inf_m[1] && y < sup_m[1]) &&
              (z >= inf_m[2] && z < sup_m[2])) {
            particles_m[is].set(total_particles_counter, w, x, y, z, mx, my, mz);
            total_particles_counter += 1;
          }
        }
      }

      // We resize the particles according to the real initialized number
      particles_m[is].resize(total_particles_counter);

      // Copy data initialized on host to device (if exist)
      particles_m[is].sync(minipic::host, minipic::device);

    } // end for species

    // initializations steps that require to use operators
    managers::initialize(params, em_m, particles_m);

    // For each species, print :
    // - total number of particles
    for (size_t is = 0; is < params.species_names_m.size(); ++is) {
      unsigned int total_number_of_particles = 0;
      double total_particle_energy           = 0;

      total_number_of_particles += particles_m[is].size();
      total_particle_energy +=
        particles_m[is].get_kinetic_energy(minipic::host);

      std::cout << " Species " << params.species_names_m[is] << std::endl;

      const double memory_consumption = total_number_of_particles * 14. * 8. / (1024. * 1024);

      std::cout << " - Initialized particles: " << total_number_of_particles << std::endl;
      std::cout << " - Total kinetic energy: " << total_particle_energy << std::endl;
      std::cout << " - Memory footprint: " << memory_consumption << " Mb" << std::endl;
    }

    // Checksum for field

    auto sum_Ex_on_host = operators::sum_power(em_m.Ex_m, 1);
    auto sum_Ey_on_host = operators::sum_power(em_m.Ey_m, 1);
    auto sum_Ez_on_host = operators::sum_power(em_m.Ez_m, 1);

    auto sum_Bx_on_host = operators::sum_power(em_m.Bx_m, 1);
    auto sum_By_on_host = operators::sum_power(em_m.By_m, 1);
    auto sum_Bz_on_host = operators::sum_power(em_m.Bz_m, 1);

    auto sum_Ex_on_device = operators::sum_power(em_m.Ex_m, 1);
    auto sum_Ey_on_device = operators::sum_power(em_m.Ey_m, 1);
    auto sum_Ez_on_device = operators::sum_power(em_m.Ez_m, 1);

    auto sum_Bx_on_device = operators::sum_power(em_m.Bx_m, 1);
    auto sum_By_on_device = operators::sum_power(em_m.By_m, 1);
    auto sum_Bz_on_device = operators::sum_power(em_m.Bz_m, 1);

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

      for (size_t is = 0; is < params.species_names_m.size(); ++is) {
        sum_host[0] += operators::sum_host(particles_m[is].weight_h_m);
        sum_host[1] += operators::sum_host(particles_m[is].x_h_m);
        sum_host[2] += operators::sum_host(particles_m[is].y_h_m);
        sum_host[3] += operators::sum_host(particles_m[is].z_h_m);
        sum_host[4] += operators::sum_host(particles_m[is].mx_h_m);
        sum_host[5] += operators::sum_host(particles_m[is].my_h_m);
        sum_host[6] += operators::sum_host(particles_m[is].mz_h_m);
        sum_host[7] += operators::sum_host(particles_m[is].Ex_h_m);
        sum_host[8] += operators::sum_host(particles_m[is].Ey_h_m);
        sum_host[9] += operators::sum_host(particles_m[is].Ez_h_m);
        sum_host[10] += operators::sum_host(particles_m[is].Bx_h_m);
        sum_host[11] += operators::sum_host(particles_m[is].By_h_m);
        sum_host[12] += operators::sum_host(particles_m[is].Bz_h_m);

        sum_device[0] += operators::sum_device(particles_m[is].weight_m);
        sum_device[1] += operators::sum_device(particles_m[is].x_m);
        sum_device[2] += operators::sum_device(particles_m[is].y_m);
        sum_device[3] += operators::sum_device(particles_m[is].z_m);
        sum_device[4] += operators::sum_device(particles_m[is].mx_m);
        sum_device[5] += operators::sum_device(particles_m[is].my_m);
        sum_device[6] += operators::sum_device(particles_m[is].mz_m);
        sum_device[7] += operators::sum_device(particles_m[is].Ex_m);
        sum_device[8] += operators::sum_device(particles_m[is].Ey_m);
        sum_device[9] += operators::sum_device(particles_m[is].Ez_m);
        sum_device[10] += operators::sum_device(particles_m[is].Bx_m);
        sum_device[11] += operators::sum_device(particles_m[is].By_m);
        sum_device[12] += operators::sum_device(particles_m[is].Bz_m);

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
    managers::iterate(params, em_m, particles_m, it);
  }

  // ________________________________________________________________
  //! \brief Perform all diagnostics
  //! \param[in] Params&  global parameters
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

    for (auto particle_binning : params.particle_binning_properties_m) {
      if (!(it % particle_binning.period_m)) {
        for (auto is : particle_binning.species_indexes_m) {
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
          particles_m[is].sync(minipic::device, minipic::host);
      }
    }

    delete[] need_species;

    if (!(it % params.field_diagnostics_period)) {
      em_m.sync(minipic::device, minipic::host);
    }

    // __________________________________________________________________
    // Start diagnostics

    // Particle binning
    for (auto particle_binning : params.particle_binning_properties_m) {

      // for each species index of this diagnostic
      for (auto is : particle_binning.species_indexes_m) {

        if (!(it % particle_binning.period_m)) {

          // Call the particle binning function using the properties in particle_binning
          Diags::particle_binning(particle_binning.name_m,
                                  params,
                                  particles_m[is],
                                  particle_binning.projected_parameter_m,
                                  particle_binning.axis_m,
                                  particle_binning.n_cells_m,
                                  particle_binning.min_m,
                                  particle_binning.max_m,
                                  is,
                                  it,
                                  particle_binning.format_m,
                                  false);

        } // end if test it % period
      }
    } // end loop on particle_binning_properties_m

    // Particle Clouds
    if ((params.particle_cloud_period < params.n_it) &&
        (!(it % params.particle_cloud_period) or (it == 0))) {

      for (size_t is = 0; is < params.get_species_number(); ++is) {
        Diags::particle_cloud("cloud", params, particles_m[is], is, it, params.particle_cloud_format);
      }
    }

    // Field diagnostics
    if (!(it % params.field_diagnostics_period)) {
      Diags::fields(params, em_m, it, params.field_diagnostics_format);
    }

    // Scalars diagnostics
    if (!(it % params.scalar_diagnostics_period)) {
      for (size_t is = 0; is < params.get_species_number(); ++is) {
        Diags::scalars(params, particles_m[is], is, it);
      }
    }

    if (!(it % params.scalar_diagnostics_period)) {
      {
        Diags::scalars(params, em_m, it);
      }
    }

  } // end diagnostics

  // __________________________________________________________________
  //
  //! \brief get the total number of particles
  // __________________________________________________________________
  unsigned int get_total_number_of_particles() {
    unsigned int total_number_of_particles = 0;

    for (size_t is = 0; is < particles_m.size(); ++is) {
      total_number_of_particles += particles_m[is].size();
    }

    return total_number_of_particles;
  }

}; // end class

/* _____________________________________________________________________ */
//! \file Params.cpp

//! \brief Methods of the class Params

/* _____________________________________________________________________ */

#include "Params.hpp"

// _____________________________________________________
//! \brief Add a particle binning object to the vector
//! of particle binning properties
//! \param[in] diag_name - string used to initiate the diag name
//! \param[in] projected_parameter - property projected on the grid, can be
//! `weight`, `density` or `particles`
//! \param[in] axis - axis to use for the grid. The axis vector size
//! determines the dimension of the diag. Axis can be `gamma`, `weight`, `x`,
//! `y`, `z`, `px`, `py` or `pz`
//! \param[in] n_cells - number of cells for each axis
//! \param[in] min - min value for each axis
//! \param[in] max - max value for each axis
//! \param[in] is - species
//! \param[in] period - output period
//! \param[in] format - (optional argument) - determine the output format, can
//! be `binary` (default) or `vtk`
// _____________________________________________________
void Params::add_particle_binning(const std::string& diag_name,
                                  const std::string& projected_parameter,
                                  const std::vector<std::string>& axis,
                                  const std::vector<std::size_t>& n_cells,
                                  const std::vector<double>& min,
                                  const std::vector<double>& max,
                                  const std::vector<int>& species_indexes,
                                  int period,
                                  const std::string& format) {

  // Check that the number of axis is the same as the number of cells, min and max
  if (axis.size() != n_cells.size() or axis.size() != min.size() or axis.size() != max.size()) {
    ERROR("ERROR in the particle binning creation: The number of axis is not the same as "
          << "the number of cells, min and max " << std::endl
          << "       axis.size() = " << axis.size() << std::endl
          << "       n_cells.size() = " << n_cells.size() << std::endl
          << "       min.size() = " << min.size() << std::endl
          << "       max.size() = " << max.size() << std::endl);
    std::raise(SIGABRT);
  }

  // Check that the species indexes exist
  for (std::size_t i = 0; i < species_indexes.size(); i++) {
    if (species_indexes[i] >= static_cast<int>(species_names_m.size())) {
      ERROR("ERROR in the particle binning creation: The species index "
            << species_indexes[i] << " does not exist" << std::endl);
      std::raise(SIGABRT);
    }
  }

  // Check that the number of cells is not zero
  for (std::size_t i = 0; i < n_cells.size(); i++) {
    if (n_cells[i] == 0) {
      ERROR("ERROR in the particle binning creation: The number of cells is zero" << std::endl);
      std::raise(SIGABRT);
    }
  }

  // Add the diag in the list
  particle_binning_properties_m.push_back(ParticleBinningProperties(diag_name,
                                                                   projected_parameter,
                                                                   axis,
                                                                   n_cells,
                                                                   min,
                                                                   max,
                                                                   species_indexes,
                                                                   period,
                                                                   format));
}

// _____________________________________________________
//
//! \brief Add a single particle to the corresponding species
//! \param[in] is species index
//! \param[in] w  particle weight
//! \param[in] x  particle weight
//! \param[in] y  particle weight
//! \param[in] z  particle weight
//! \param[in] mx  particle weight
//! \param[in] my  particle weight
//! \param[in] mz  particle weight
// _____________________________________________________
void Params::add_particle(unsigned int is,
                          double w,
                          double x,
                          double y,
                          double z,
                          double mx,
                          double my,
                          double mz) {

  particles_to_add_m.push_back(Particle(is, w, x, y, z, mx, my, mz));
}

// _________________________________________________________________________________________________
//! \brief Add an antenna
//! \param antenna_profile Lambda std::function<double(double, double, double)> representing the
//! antenna profile
//! \param x Position of the antenna
// _________________________________________________________________________________________________
void Params::add_antenna(std::function<double(double, double, double)> antenna_profile, double x) {
  antenna_profiles_m.push_back(antenna_profile);
  antenna_positions_m.push_back(x);
}

// _________________________________________________________________________________________________
//! \brief Add an function imbalance
//! \param imbalance_profile add the imbalance function
// _________________________________________________________________________________________________
void Params::add_imbalance(std::function<double(double, double, double, double)> function_profile) {
  imbalance_function_m.push_back(function_profile);
}

// _________________________________________________________________________________________________
//! \brief Add a species
//! \param name Name of the species
//! \param mass Mass of the species
//! \param charge Charge of the species
//! \param temp Temperature of the species
//! \param density_profile Lambda std::function<double(double, double, double)> representing the
//! density profile of the species \param drift_velocity Drift velocity of the species
//! \param ppc Number of particles per cell of the species
//! \param position_initiatization_method Method to use for position init
//! \param position_initiatization_level Level to use for position init
// _________________________________________________________________________________________________
void Params::add_species(const std::string& name,
                         double mass,
                         double charge,
                         double temp,
                         std::function<double(double, double, double)> density_profile,
                         const std::vector<double>& drift_velocity,
                         double ppc,
                         const std::string& position_initiatization_method,
                         const std::string& position_initiatization_level) {

  // Check that the drift velocity is below the speed of light
  const double v = drift_velocity[0] * drift_velocity[0] + drift_velocity[1] * drift_velocity[1] +
                   drift_velocity[2] * drift_velocity[2];
  if (v > 1) {
    ERROR(" The drift velocity of species " << name << " is above the speed of light (v > 1)"
                                            << std::endl);
    std::raise(SIGABRT);
  }

  species_names_m.push_back(name);
  ppc_m.push_back(ppc);
  density_profiles_m.push_back(density_profile);
  mass_m.push_back(mass);
  charge_m.push_back(charge);
  temp_m.push_back(temp);
  position_initialization_method_m.push_back(position_initiatization_method);
  position_initialization_level_m.push_back(position_initiatization_level);
  drift_velocity_m.push_back(drift_velocity);
}

// _____________________________________________________
//
//! \brief Set the intial values of the electric field
//! \param[in] Ex initial value for Ex
//! \param[in] Ez initial value for Ey
//! \param[in] Ey initial value for Ez
// _____________________________________________________
void Params::initialize_electric_field(double Ex, double Ey, double Ez) {

  E0_m[0] = Ex;
  E0_m[1] = Ey;
  E0_m[2] = Ez;
}

// _____________________________________________________
//
//! \brief Set the intial values of the magnetic field
//! \param[in] Bx initial value for Bx
//! \param[in] By initial value for By
//! \param[in] Bz initial value for Bz
// _____________________________________________________
void Params::initialize_magnetic_field(double Bx, double By, double Bz) {
  B0_m[0] = Bx;
  B0_m[1] = By;
  B0_m[2] = Bz;
}

// ___________________________________________________________
//
//! \brief Simple parser to read some input parameters
//! from command line arguments
// ___________________________________________________________
void Params::read_from_command_line_arguments(int argc, char *argv[]) {

  // convert argv in list of string
  std::vector<std::string> args(argv, argv + argc);

  if (argc > 1) {
    int iarg = 1;
    while (iarg < argc) {

      const std::string key(args[iarg]);
      if (key == "-it" or key == "--iterations") {
        const unsigned int iterations = std::stoi(args[iarg + 1]);
        simulation_time               = iterations * dt;
        iarg += 2;
      } else if (key == "-cpp" or key == "--cells") {
        nx_cells = std::stoi(args[iarg + 1]);
        ny_cells = std::stoi(args[iarg + 2]);
        nz_cells = std::stoi(args[iarg + 3]);
        iarg += 4;
      } else if (key == "-dmin" or key == "--domain_min") {
        inf_x = std::stod(args[iarg + 1]);
        inf_y = std::stod(args[iarg + 2]);
        inf_z = std::stod(args[iarg + 3]);
        iarg += 4;
      } else if (key == "-dmax" or key == "--domain_max") {
        sup_x = std::stod(args[iarg + 1]);
        sup_y = std::stod(args[iarg + 2]);
        sup_z = std::stod(args[iarg + 3]);
        iarg += 4;
      } else if (key == "-ppc" or key == "--particles_per_cell") {
        const unsigned int ppc = std::stoi(args[iarg + 1]);
        for (unsigned int i = 0; i < ppc_m.size(); i++)
          ppc_m[i] = ppc;
        iarg += 2;
      }

      else if (key == "-rs" or key == "--random_seed") {
        seed = std::stoi(args[iarg + 1]);
        iarg += 2;
      } else if (key == "-pp" or key == "--print_period") {
        print_period = std::stoi(args[iarg + 1]);
        iarg += 2;
      } else if (key == "-stp" or key == "--save_timers_period") {
        save_timers_period = std::stoi(args[iarg + 1]);
        iarg += 2;
      } else if (key == "-sts" or key == "--save_timers_start") {
        save_timers_start = std::stoi(args[iarg + 1]);
        iarg += 2;
      } else if (key == "-h" or key == "--help") {
        help();
        iarg++;
      } else if (key.find("threads") != std::string::npos) {
        iarg++;
      } else if (key.find("kokkos") != std::string::npos) {
        iarg++;
        // Skip next arg if not a flag
        if (iarg < argc && (argv[iarg][0] != '-')) {
          iarg++;
        }
      } else {
        std::cerr << " Argument " << argv[iarg] << " is not recognized." << std::endl;
        std::cerr << " Use argument `-h` for help." << std::endl;
        std::raise(SIGABRT);
      }
    }
  }
}

// _____________________________________________________
//
//! \brief print the title
// _____________________________________________________
void Params::title() {

  std::cout << seperator(50) << std::endl;
  std::cout << std::endl;
   
  std::cout << "             o          o                              \n"
            << "   _  _  _       _  _                                  \n"
            << "  / |/ |/ |  |  / |/ |  |  _       _       _           \n"
            << "    |  |  |_/|_/ _|_ |_/|_/_)_ __ (_)_ __ (_) ___      \n"
            << "                | '_ ` _ \\| | '_ \\| | '_ \\| |/ __|  \n"
            << "                | | | | | | | | | | | |_) | | (__      \n"
            << "                |_| |_| |_|_|_| |_|_| .__/|_|\\___|    \n"
            << "                                    |_|                \n"
            << std::endl;

  std::cout << seperator(50) << std::endl;
  std::cout << std::endl;

  DEBUG("Debug mode activated");

}

// _____________________________________________________
//
//! \brief Print input parameters summary
// _____________________________________________________
void Params::info() {

  std::cout << " Simulation name: " << name << std::endl;
  std::cout << std::endl;

  std::cout << " Input parameters summary: " << std::endl;

  std::cout << std::endl;
  std::cout << " > Time: " << std::endl;
  std::cout << "   - simulation time: " << simulation_time << std::endl;
  std::cout << "   - dt: " << dt << std::endl;
  std::cout << "   - CFL: " << dt_cfl << std::endl;
  std::cout << "   - number of iterations: " << n_it << std::endl;

  std::cout << std::endl;
  std::cout << " > Domain: " << std::endl;
  std::cout << "   - min/max in x: " << inf_x << " " << sup_x << std::endl;
  std::cout << "   - min/max in y: " << inf_y << " " << sup_y << std::endl;
  std::cout << "   - min/max in z: " << inf_z << " " << sup_z << std::endl;
  std::cout << "   - Boundary conditions: " << boundary_condition << std::endl;

  std::cout << std::endl;
  std::cout << "   - cells: " << nx_cells << " " << ny_cells << " " << nz_cells << std::endl;
  std::cout << "   - space step: " << dx << " " << dy << " " << dz << std::endl;

  std::cout << std::endl;
  for (std::size_t is = 0; is < get_species_number(); ++is) {

    std::cout << " > Species " << is << ": " << species_names_m[is] << "\n"
              << "   - mass: " << mass_m[is] << "\n"
              << "   - charge: " << charge_m[is] << "\n"
              << "   - particles per cell: " << ppc_m[is] << "\n"
              << "   - temperature: " << temp_m[is] << "\n"
              << "   - drift velocity: " << drift_velocity_m[is][0] << " " << drift_velocity_m[is][1]
              << " " << drift_velocity_m[is][2] << "\n"
              << "   - pos init method: " << position_initialization_method_m[is] << "\n"
              << "   - pos init level: " << position_initialization_level_m[is] << "\n"
              << std::endl;
  }

  if (!current_projection or !maxwell_solver) {
    std::cout << " > Operators: " << std::endl;
    if (!current_projection) {
      std::cout << "   - Current projection disabled " << std::endl;
    }
    if (!maxwell_solver) {
      std::cout << "   - Maxwell Solver disabled " << std::endl;
    }
    std::cout << std::endl;
  }

  std::cout << " > Memory usage: " << std::endl;
  const double EM_grid_size =
    (nx_cells + 2) * (ny_cells + 2) * (nz_cells + 2) * 8 / 1024.;
  const double patch_current_grid_size =
    (nx_cells + 4) * (ny_cells + 4) * (nz_cells + 4) * 8 / 1024.;
  std::cout << "   - EM grid size: " << EM_grid_size << " Kb" << std::endl;
  std::cout << "   - Current grid size: " << patch_current_grid_size << " Kb" << std::endl;
  std::cout << "   - Total grid size: "
            << EM_grid_size * 6 + patch_current_grid_size * 3 << " Kb" << std::endl;
  std::cout << std::endl;

  std::cout << " > Timers: " << std::endl;
  std::cout << "   - save timers period: " << save_timers_period << std::endl;
  std::cout << "   - save timers start: " << save_timers_start << std::endl;
  std::cout << "   - Bufferize timers: " << bufferize_timers_output << std::endl;
  std::cout << std::endl;

  std::cout << " > General diag properties: " << std::endl;
  if (no_diagnostics_at_init) {
    std::cout << "   - Diagnostics disabled at initialization" << std::endl;
  }
  std::cout << "   - output directory: " << output_directory << std::endl;
  if (particle_cloud_period > 0 and particle_cloud_period < n_it) {
    std::cout << "   - cloud output period: " << particle_cloud_period << std::endl;
  }
  std::cout << "   - field output period: " << field_diagnostics_period << std::endl;
  std::cout << "   - scalar output period: " << scalar_diagnostics_period << std::endl;
  std::cout << std::endl;

  auto N = get_particle_binning_number();

  if (N > 0) {
    std::cout << " > Particle binning: " << std::endl;
    for (std::size_t id = 0; id < N; ++id) {
      auto dim = particle_binning_properties_m[id].axis_m.size();
      std::cout << "   - " << particle_binning_properties_m[id].name_m << " on species ";
      for (auto is : particle_binning_properties_m[id].species_indexes_m) {
        std::cout << is << " ";
      }
      std::cout << "\n";
      for (std::size_t d = 0; d < dim; ++d) {
        std::cout << "     * axis " << d << ": " << particle_binning_properties_m[id].axis_m[d]
                  << " [" << particle_binning_properties_m[id].min_m[d] << ", "
                  << particle_binning_properties_m[id].max_m[d] << ", "
                  << particle_binning_properties_m[id].n_cells_m[d] << "]"
                  << "\n";
      }
      std::cout << "     * format: " << particle_binning_properties_m[id].format_m << "\n";
      std::cout << "     * output period: " << particle_binning_properties_m[id].period_m << "\n";
    }
  }

  std::cout << std::endl;
}

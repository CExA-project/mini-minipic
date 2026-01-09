/* _____________________________________________________________________ */
//! \file Params.hpp

//! \brief description of the class Params that contains all
//! global simulation parameters

/* _____________________________________________________________________ */

#pragma once

#include "Headers.hpp"
#include "Random.hpp"
#include "Tools.hpp"
#include <csignal>
#include <cstdio>
#include <functional>
#include <iostream>
#include <cmath>
#include <vector>

// ___________________________________________________________________
//
//! structure to store the properties of a particle binning diagnostics
// ___________________________________________________________________
struct ParticleBinningProperties {
  //! Name of the diagnostic (used for output)
  std::string name_m;
  //! Name of the parameter to project
  std::string projected_parameter_m;
  //! vector of axis names
  std::vector<std::string> axis_m;
  //! Number of cells for each axis
  std::vector<std::size_t> n_cells_m;
  //! min values for each axis
  std::vector<double> min_m;
  //! max values for each axis
  std::vector<double> max_m;
  //! Species number
  std::vector<int> species_indexes_m;
  //! Output period
  int period_m;
  //! format of the output file
  std::string format_m;

  //! Constructor
  ParticleBinningProperties(const std::string& name,
                            const std::string& projected_parameter,
                            const std::vector<std::string>& axis,
                            const std::vector<std::size_t>& n_cells,
                            const std::vector<double>& min,
                            const std::vector<double>& max,
                            const std::vector<int>& species_indexes,
                            int period,
                            const std::string& format)
    : name_m(name), projected_parameter_m(projected_parameter), axis_m(axis), n_cells_m(n_cells),
      min_m(min), max_m(max), species_indexes_m(species_indexes), period_m(period), format_m(format) {};
};

// ___________________________________________________________________
//
//! structure to store a single particle
// ___________________________________________________________________
struct Particle {

  unsigned int is_m;
  double weight_m;
  double x_m;
  double y_m;
  double z_m;
  double mx_m;
  double my_m;
  double mz_m;

  //! Constructor
  Particle(unsigned int is, double w, double x, double y, double z, double mx, double my, double mz)
    : is_m(is), weight_m(w), x_m(x), y_m(y), z_m(z), mx_m(mx), my_m(my), mz_m(mz) {};
};

// ___________________________________________________________________
//
//! Structure to store all input parameters and those that can be computed
// ___________________________________________________________________
class Params {
public:
  Params() {}
  ~Params() {}

  // Simulation name
  std::string name;

  //      Discretisation param
  // Space
  //! Domain boundaries
  double inf_x, inf_y, inf_z;
  double sup_x, sup_y, sup_z;

  //! Input computed
  std::size_t nx_cells, ny_cells, nz_cells;
  double Lx, Ly, Lz;
  double dx, dy, dz;
  double inv_dx, inv_dy, inv_dz;
  double dx_sq, dy_sq, dz_sq;
  double cell_volume, inv_cell_volume;

  int nx_p, ny_p, nz_p;
  int nx_d, ny_d, nz_d;

  //! String to store the name of the boundary condition
  std::string boundary_condition;

  //! Code to shortcut the type of boundary conditions
  //! 0 - free
  //! 1 - periodic
  //! 2 - reflective
  int boundary_condition_code;

  //! Iteration period for information display in the terminal
  int print_period;

  // Time
  //! Start, stop and step for the simulation time
  double dt;
  //! Time total, computed
  double simulation_time;
  //! Number of iteration, computed
  std::size_t n_it;
  //! CFL
  double dt_cfl;

  // ____________________________________________________________
  // Species parameters

  //! Name of the species
  std::vector<std::string> species_names_m;
  //! Particle per cell for each species at init
  std::vector<std::size_t> ppc_m;
  //! Normalized density, temperature, mass, charge for each species at init
  std::vector<double> temp_m, mass_m, charge_m;
  //! Density profile for each species at init
  //! Lambda of 3 doubles: x, y, z that representes a normalized position (between 0 and 1)
  std::vector<std::function<double(double, double, double)>> density_profiles_m;
  //! Mean velocity for each species at init (vector of 3 doubles)
  std::vector<std::vector<double>> drift_velocity_m;
  //! Method to use for position init
  std::vector<std::string> position_initialization_method_m;
  //! Position init level (patch or cell)
  std::vector<std::string> position_initialization_level_m;

  //! Number of particles total for each species at init, computed
  std::vector<std::size_t> n_particles_by_species;
  //! Number of particles at init, computed
  std::size_t n_particles;

  //! list of particles to add at init
  std::vector<Particle> particles_to_add_m;

  // ____________________________________________________________
  // Antenna parameters

  //! Antenna profile (x, y, t)
  std::vector<std::function<double(double, double, double)>> antenna_profiles_m;
  //! Antenna position
  std::vector<double> antenna_positions_m;

  // ____________________________________________________________
  // Imbalance parameters

  //! imbalance profile (x, y, z)
  std::vector<std::function<double(double, double, double, double)>> imbalance_function_m;

  //! Random number
  int seed = 0;

  // ____________________________________________________________
  // Initial Electric and magnetic field values

  double E0_m[3] = {0, 0, 0};
  double B0_m[3] = {0, 0, 0};

  // ____________________________________________________________
  // Operators

  //! Momentum correction
  bool momentum_correction = false;

  //! Projector
  bool current_projection = true;

  //! Projector
  bool maxwell_solver = true;

  // ____________________________________________________________
  // Parallelism

  bool on_gpu_m = false;

  // ____________________________________________________________
  // Diagnostics parameters

  //! If true, no diagnostics at iteration 0
  bool no_diagnostics_at_init = false;

  //! Output directory relative path
  std::string output_directory = "diags";

  //! vector of ParticleBinningProperties
  std::vector<ParticleBinningProperties> particle_binning_properties_m;

  //! Period of the particle cloud diagnostic
  unsigned int particle_cloud_period = 0;

  //! Format of the particle cloud diagnostic
  std::string particle_cloud_format = "binary";

  //! Period of the field diagnostic
  unsigned int field_diagnostics_period = 0;

  //! Format of the field diagnostic
  std::string field_diagnostics_format = "vtk";

  //! Period for scalar diagnostics
  unsigned int scalar_diagnostics_period = 0;

  //! Digits for iteration in output file names
  unsigned int max_it_digits = 0;

  //! Period for timers saving
  unsigned int save_timers_period = 0;

  //! start for timers saving
  unsigned int save_timers_start = 0;

  //! bufferize the output
  bool bufferize_timers_output = true;

  // _________________________________________
  // Methods

  // _______________________________________________________________
  //
  //! \brief Compute all global parameters using the user input
  //! Check that the input parameters are coherent
  // _______________________________________________________________
  void compute() {

    DEBUG("Compute global parameters");

    Lx = sup_x - inf_x;
    Ly = sup_y - inf_y;
    Lz = sup_z - inf_z;

    dx = Lx / nx_cells;
    dy = Ly / ny_cells;
    dz = Lz / nz_cells;

    inv_dx = 1. / dx;
    inv_dy = 1. / dy;
    inv_dz = 1. / dz;

    dx_sq = dx * dx;
    dy_sq = dy * dy;
    dz_sq = dz * dz;

    cell_volume     = dx * dy * dz;
    inv_cell_volume = inv_dx * inv_dy * inv_dz;

    nx_p = nx_cells + 1;
    ny_p = ny_cells + 1;
    nz_p = nz_cells + 1;
    nx_d = nx_cells + 2;
    ny_d = ny_cells + 2;
    nz_d = nz_cells + 2;

    // boundary conditions

    if (boundary_condition == "periodic") {
      boundary_condition_code = 1;
    } else if (boundary_condition == "reflective") {
      boundary_condition_code = 2;
    } else {
      ERROR(" Boundary condition " << boundary_condition << " is not supported");
      std::raise(SIGABRT);
    }

    // Time

    // Compute the exacte CFL condition
    dt_cfl = std::sqrt(1 / (1 / dx_sq + 1 / dy_sq + 1 / dz_sq));

    if (dt > 1 || dt <= 0) {
      ERROR("ERROR in setup: dt (fraction of the CFL) must be between 0 and 1 to comply with the "
            "CFL condition")
      std::raise(SIGABRT);
    }

    // Get the number of time steps
    n_it = static_cast<int>(std::round(simulation_time / dt));

    // Convert dt into a fraction of the CFL
    dt              = dt * dt_cfl;
    simulation_time = n_it * dt;

    // const double cfl = (dt * dt / (1 / dx_sq + 1 / dy_sq + 1 / dz_sq));
    // if (cfl > 1) {
    //   std::cerr << " CFL condition is not respected, you must have : 1/dt**2 <= 1/dx**2 + 1/dy**2
    //   "
    //                "+ 1/dz**2 "
    //             << std::endl;
    //   std::raise(SIGABRT);
    // }

    // Physics
    n_particles = 0;
    n_particles_by_species.resize(species_names_m.size());
    for (std::size_t is = 0; is < species_names_m.size(); is++) {
      n_particles_by_species[is] =nx_cells * ny_cells * nz_cells * ppc_m[is];
      n_particles += n_particles_by_species[is];
    }

    // Check species initialization
    for (std::size_t is = 0; is < species_names_m.size(); ++is) {

      bool passed = false;

      if (position_initialization_method_m[is] == "random") {

        passed = true;

      } else {

        // We check that the position init is one of the previous species
        for (std::size_t is2 = 0; is2 < is; ++is2) {
          if (position_initialization_method_m[is] == species_names_m[is2]) {
            passed = true;
          }
        }
      }
      // if not passed, return an error
      if (!passed) {
        ERROR(" Position initialization " << position_initialization_method_m[is]
                                          << " is not supported");
        std::raise(SIGABRT);
      }
    }

    // Check species init level (should be "cell" or "patch")
    for (std::size_t is = 0; is < species_names_m.size(); ++is) {
      if (position_initialization_level_m[is] != "cell" &&
          position_initialization_level_m[is] != "patch") {
        ERROR(" Position initialization level " << position_initialization_level_m[is]
                                                << " is not supported");
        std::raise(SIGABRT);
      }
    }

    // _________________________________________
    // Diagnostics

    // get number of digit for the number of iterations
    // Used for diagnostics names
    max_it_digits = 0;
    std::size_t n_it_tmp  = n_it;
    while (n_it_tmp > 0) {
      n_it_tmp /= 10;
      max_it_digits++;
    }

    // if the period is 0, then no outputs
    if (particle_cloud_period == 0) {
      particle_cloud_period = n_it + 1;
    }

    if (field_diagnostics_period == 0) {
      field_diagnostics_period = n_it + 1;
    }

    if (scalar_diagnostics_period == 0) {
      scalar_diagnostics_period = n_it + 1;
    }

    if (save_timers_period == 0) {
      save_timers_period = n_it + 1;
    }

    // _________________________________________
    // Parallelism

    DEBUG("End of compute global parameters");

  }

  // _________________________________________________________________________________________________
  //! \brief Add an antenna
  //! \param antenna_profile Lambda std::function<double(double, double, double)> representing the
  //! antenna profile
  //! \param x Position of the antenna
  // _________________________________________________________________________________________________
  void add_antenna(std::function<double(double, double, double)> antenna_profile, double x);

  // _________________________________________________________________________________________________
  //! \brief Add imbalance function
  //! \param function_profile function
  // _________________________________________________________________________________________________
  void add_imbalance(std::function<double(double, double, double, double)> function_profile);

  // _________________________________________________________________________________________________
  //! \brief Add a species
  //! \param name Name of the species
  //! \param mass Mass of the species
  //! \param charge Charge of the species
  //! \param temp Temperature of the species
  //! \param density_profile Lambda std::function<double(double, double, double)> representing the
  //! density profile of the species \param drift_velocity Drift velocity of the species \param ppc
  //! Number of particles per cell of the species \param position_initiatization Method to use for
  //! position init
  // _________________________________________________________________________________________________
  void add_species(const std::string& name,
                   double mass,
                   double charge,
                   double temp,
                   std::function<double(double, double, double)> density_profile,
                   const std::vector<double>& drift_velocity,
                   double ppc,
                   const std::string& position_initiatization,
                   const std::string& position_initialization_level);

  // _____________________________________________________
  //
  //! \brief Add a particle binning object (diagnostic) to the vector
  //! of particle binning properties
  //! \param[in] diag_name - string used to initiate the diag name
  //! \param[in] projected_parameter - property projected on the grid, can be
  //! `weight`
  //! \param[in] axis - axis to use for the grid. The axis vector size
  //! determines the dimension of the diag. Axis can be `gamma`, `weight`, `x`,
  //! `y`, `z`, `px`, `py`, `pz`
  //! \param[in] n_cells - number of cells for each axis
  //! \param[in] min - min value for each axis
  //! \param[in] max - max value for each axis
  //! \param[in] is - species
  //! \param[in] period - output period
  //! \param[in] format - (optional argument) - determine the output format, can
  //! be `binary` (default) or `vtk`
  // _____________________________________________________
  void add_particle_binning(const std::string& diag_name,
                            const std::string& projected_parameter,
                            const std::vector<std::string>& axis,
                            const std::vector<std::size_t>& n_cells,
                            const std::vector<double>& min,
                            const std::vector<double>& max,
                            const std::vector<int>& species_indexes,
                            int period,
                            const std::string& format = "binary");

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
  void add_particle(unsigned int is,
                    double w,
                    double x,
                    double y,
                    double z,
                    double mx,
                    double my,
                    double mz);

  // _____________________________________________________
  //
  //! \brief Set the intial values of the electric field
  //! \param[in] Ex initial value for Ex
  //! \param[in] Ez initial value for Ey
  //! \param[in] Ez initial value for Ez
  // _____________________________________________________
  void initialize_electric_field(double Ex, double Ey, double Ez);

  // _____________________________________________________
  //
  //! \brief Set the intial values of the magnetic field
  //! \param[in] Bx initial value for Bx
  //! \param[in] By initial value for By
  //! \param[in] Bz initial value for Bz
  // _____________________________________________________
  void initialize_magnetic_field(double Bx, double By, double Bz);

  // _____________________________________________________
  //
  //! \brief return the number of species
  // _____________________________________________________
  inline auto get_species_number() const { return species_names_m.size(); }

  // _____________________________________________________
  //
  //! \brief return the number of particle binning objects
  // _____________________________________________________
  inline auto get_particle_binning_number() const { return particle_binning_properties_m.size(); }

  // _____________________________________________________
  //
  //! \brief print the help for command line options
  // _____________________________________________________
  void help() const {
    std::cout
      << " \n"
      << " Help for command line options \n"
      << " Note: command line options overwrite program parameters.\n\n"
      << " -h   (--help): print the help page for command line options\n"
      << " -it  (--iterations) int: change the number of iterations\n"
      << " -dmin (--domain_min) double double double: change the domain minimum boundaries\n"
      << " -dmax (--domain_max) double double double: change the domain maximum boundaries\n"
      << " -p   (--patches) int int int: number of patches per direction\n"
      << " -cpp (--cells_per_patch) int int int: number of cells per patch per direction\n"
      << " -rs  (--random_seed) int: seed for random generator\n"
      << " -pp  (--print_period) int: iteration period for terminal printing\n"
      << " -stp (--save_timers_period) int: iteration period for timers saving\n"
      << " -sts (--save_timers_start) int: iteration start for timers saving\n"
      << std::endl;
    std::_Exit(EXIT_SUCCESS);
  }

  // _____________________________________________________
  //
  //! \brief Simple parser to read some input parameters
  //! from command line arguments
  // _____________________________________________________
  void read_from_command_line_arguments(int argc, char *argv[]);

  // _____________________________________________________
  //
  //! \brief Create a line composed of `size` characters
  //! \param[in] size - number of characters for a single line
  // _____________________________________________________
  std::string seperator(int size) const {
    std::string line = " ";
    for (int i = 0; i < size; ++i) {
      line += "_";
    }
    return line;
  }

  // _____________________________________________________
  //
  //! \brief print the title
  // _____________________________________________________
  void title();

  // _____________________________________________________
  //
  //! \brief Print input parameters summary
  // _____________________________________________________
  void info();
};

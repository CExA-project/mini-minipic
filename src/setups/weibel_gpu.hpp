/* _____________________________________________________________________ */
//! \file weibel.hpp

//! \brief This is a setup used to initialize MiniPic
//! CASE: Weibel like instabilities suing 2 colliding plasma with periodic boundary conditions

/* _____________________________________________________________________ */

#include "Params.hpp"

//! \brief Functiun to setup all input parameters
void setup(Params &params) {

  // Simulation name
  params.name = "Weibel instability";

  // Physics parameters
  const double v_drift      = 0.5;
  const double temperature  = 0.1 / 511;
  const double n0           = 1.;
  const double debye_length = std::sqrt(temperature / n0);

  const double dx = debye_length / 8;
  const double dy = debye_length / 4;
  const double dz = debye_length / 4;

  // Space: compute the domain size from the number of cells and number of patches
  params.inf_x = 0.;
  params.inf_y = 0.;
  params.inf_z = 0.;
  params.sup_x = 2048 * dx;
  params.sup_y = 32 * dy;
  params.sup_z = 32 * dz;

  // Decomp
  params.n_subdomains = 1;

  int nx_cells = static_cast<int>((params.sup_x - params.inf_x) / dx);
  int ny_cells = static_cast<int>((params.sup_y - params.inf_y) / dy);
  int nz_cells = static_cast<int>((params.sup_z - params.inf_z) / dz);

  // Number of patches
  params.nx_patch = 1;
  params.ny_patch = 1;
  params.nz_patch = 1;

  // Cells per patch per direction
  params.nx_cells_by_patch = nx_cells / params.nx_patch;
  params.ny_cells_by_patch = ny_cells / params.ny_patch;
  params.nz_cells_by_patch = nz_cells / params.nz_patch;

  // Time
  params.dt = 0.5;

  params.simulation_time = 20;

  // Species

  // custom density profile
  auto profile_left = [](double x, double y, double z) -> double {
    if (x < 0.5) {
      return 1e-2;
    } else {
      return 0;
    }
  };
  auto profile_right = [](double x, double y, double z) -> double {
    if (x > 0.5) {
      return 1e-2;
    } else {
      return 0;
    }
  };

  // Plasma left
  params.add_species("electron_left",
                     1,
                     -1,
                     temperature,
                     profile_left,
                     {v_drift, 0, 0},
                     8,
                     "random",
                     "cell");
  params.add_species("positron_left",
                     1,
                     1,
                     temperature,
                     profile_left,
                     {v_drift, 0, 0},
                     8,
                     "electron_left",
                     "cell");
  // // Plasma right
  params.add_species("electron_right",
                     1,
                     -1,
                     temperature,
                     profile_right,
                     {-v_drift, 0, 0},
                     8,
                     "random",
                     "cell");
  params.add_species("positron_right",
                     1,
                     1,
                     temperature,
                     profile_right,
                     {-v_drift, 0, 0},
                     8,
                     "electron_right",
                     "cell");

  // Bourndary conditions
  params.boundary_condition = "periodic";

  // Display
  params.print_period = 50;

  // Random seed
  params.seed = 0;

  // Scalar Diagnostics
  params.scalar_diagnostics_period = 50;

  // Field Diagnostics
  params.field_diagnostics_period = 100;

  // Particle Diagnostics
  params.add_particle_binning("diag_x_y_z_d",
                              "density",
                              {"x", "y", "z"},
                              {params.nx_patch * params.nx_cells_by_patch,
                               params.ny_patch * params.ny_cells_by_patch,
                               params.nz_patch * params.nz_cells_by_patch},
                              {0., 0., 0.},
                              {params.sup_x, params.sup_y, params.sup_z},
                              {0, 1},
                              100,
                              "bin");
}
/* _____________________________________________________________________ */
//! \file default.hpp

//! \brief This is the default setup used to initialize MiniPic
//! CASE: homogeneous uniform thermalized plasma with periodic boundary conditions

/* _____________________________________________________________________ */

#include "Params.hpp"

//! \brief Functiun to setup all input parameters
void setup(Params &params) {

  // Physics parameters
  const double n0           = 1e-3;
  const double debye_length = sqrt(n0);
  const double v_drift      = 0.5;
  const double temperature  = debye_length * debye_length;

  const double dx = debye_length / 10;
  const double dy = debye_length / 10;
  const double dz = debye_length / 10;

  // Space: compute the domain size from the number of cells and number of patches
  params.inf_x = 0.;
  params.inf_y = 0.;
  params.inf_z = 0.;
  params.sup_x = 32 * dx;
  params.sup_y = 32 * dy;
  params.sup_z = 32 * dz;

  // Decomp
  params.n_subdomains = 1;

  int nx_cells = static_cast<int>((params.sup_x - params.inf_x) / dx);
  int ny_cells = static_cast<int>((params.sup_y - params.inf_y) / dy);
  int nz_cells = static_cast<int>((params.sup_z - params.inf_z) / dz);

  // Number of patches
  params.nx_patch = 4;
  params.ny_patch = 4;
  params.nz_patch = 4;

  // Cells per patch per direction
  params.nx_cells_by_patch = nx_cells / params.nx_patch;
  params.ny_cells_by_patch = ny_cells / params.ny_patch;
  params.nz_cells_by_patch = nz_cells / params.nz_patch;

  // Time
  params.dt = 0.5;

  params.simulation_time = 200 * params.dt;

  // Species

  // custom density profile
  auto profile = [](double x, double y, double z) -> double { return 1e-2; };

  // Plasma left
  params.add_species("electron_left",
                     1,
                     -1,
                     temperature,
                     profile,
                     {v_drift, 0, 0},
                     8,
                     "random",
                     "cell");
  params.add_species("positron_left",
                     1,
                     1,
                     temperature,
                     profile,
                     {v_drift, 0, 0},
                     8,
                     "electron_left",
                     "cell");
  // // Plasma right
  params.add_species("electron_right",
                     1,
                     -1,
                     temperature,
                     profile,
                     {-v_drift, 0, 0},
                     8,
                     "random",
                     "cell");
  params.add_species("positron_right",
                     1,
                     1,
                     temperature,
                     profile,
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
}
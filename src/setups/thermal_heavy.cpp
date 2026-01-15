/* _____________________________________________________________________ */
//! \file default.hpp

//! \brief This is the default setup used to initialize MiniPic
//! CASE: homogeneous uniform thermalized plasma with periodic boundary conditions

/* _____________________________________________________________________ */

#include "Setup.hpp"

//! \brief Functiun to setup all input parameters
void setup(Params &params) {

  // Simulation name
  params.name = "thermal heavy";

  // Space
  params.inf_x = 0.;
  params.inf_y = 0.;
  params.inf_z = 0.;
  params.sup_x = 1.;
  params.sup_y = 1.;
  params.sup_z = 1.;

  params.nx_cells = 32 * 3;
  params.ny_cells = 32 * 3;
  params.nz_cells = 32 * 3;

  params.dt = 0.9;

  params.simulation_time = 200 * params.dt;

  // Species

  // custom density profile
  auto profile = [](double, double, double) -> double {
    // if ((x > 0.2) && (x < 0.8)) {
    return 1e-5;
    // } else {
    //   return 0;
    // }
  };

  // name, mass, charge, temperature, density profile, drift velocity, particles per cell,
  // position initialization
  params.add_species("electron", 1, -1, 1e-2, profile, {0, 0, 0}, 8 * 2, "random", "cell");
  params.add_species("proton", 1836.125, 1, 1e-2, profile, {0, 0, 0}, 8 * 2, "electron", "cell");

  // Momentum correction at init
  params.momentum_correction = false;

  // Bourndary conditions
  params.boundary_condition = "periodic";

  // Display
  params.print_period = 50;

  // Random seed
  params.seed = 0;

  // Scalar Diagnostics
  params.scalar_diagnostics_period = 200;

  // Field Diagnostics
  params.field_diagnostics_period = 200;
  params.field_diagnostics_format = "vtk";

  // Cloud Diagnostics
  params.particle_cloud_period = 200;
  params.particle_cloud_format = "binary";

  // Binning Diagnostics
  params.add_particle_binning("diag_w_gamma",
                              "weight",
                              {"gamma"},
                              {32},
                              {0},
                              {0},
                              {0, 1},
                              200,
                              "binary");

  params.add_particle_binning("diag_x_y_z_d",
                              "density",
                              {"x", "y", "z"},
                              {params.nx_cells,
                               params.ny_cells,
                               params.nz_cells},
                              {0., 0., 0.},
                              {params.sup_x, params.sup_y, params.sup_z},
                              {0, 1},
                              200,
                              "vtk");

  params.add_particle_binning("diag_px_py_pz_d",
                              "density",
                              {"px", "py", "pz"},
                              {params.nx_cells,
                               params.ny_cells,
                               params.nz_cells},
                              {0., 0., 0.},
                              {0, 0, 0},
                              {0, 1},
                              200,
                              "vtk");
}

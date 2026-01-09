
/* _____________________________________________________________________ */
//! \file Diagnostics.hpp

//! \brief Diagnostics functions to output data on disks
//!
/* _____________________________________________________________________ */

#pragma once

#include <csignal>
#include <iomanip>
#include <iostream>
#include <vector>

#include "Operators.hpp"
#include "Params.hpp"
#include "Particles.hpp"

namespace Diags {

// _____________________________________________________________________
//! \brief Initialization of the diagnostics
//! -> Create the output directory
// _____________________________________________________________________
void initialize(Params &params) {
  // Create the output directory
  std::string command = "mkdir -p " + params.output_directory;
  int error_code = system(command.c_str());
  if (error_code != 0) {
    ERROR(" Error while creating the directory :" << params.output_directory)
    std::raise(SIGABRT);
  }
}

// _____________________________________________________________________
//! \brief output a structured array of any dimension in a binary file

//! WARNING: This function is not optimized
//! This function should be put in subdomain or in a specific diag class

//! \param[in] file_name - name of the file on disk
//! \param[in] projected_parameter - name of the data parameter
//! \param[in] axis - axis names
//! \param[in] n_cells number of cells for each axis
//! \param[in] min - min value for each axis
//! \param[in] max - max value for each axis
//! \param[in] data pointer to the raw data
// _____________________________________________________________________
void output_binary_structured_grid(const std::string& file_name,
                                   const std::string& projected_parameter,
                                   const std::vector<std::string>& axis,
                                   const std::vector<std::size_t>& n_cells,
                                   const std::vector<double>& min,
                                   const std::vector<double>& max,
                                   double *__restrict__ data) {

  const unsigned int dim = axis.size();
  unsigned int data_size = 1;

  std::ofstream binary_file(file_name, std::ios::out | std::ios::binary);
  // char * data_ptr = (char *) data.data();
  binary_file.write((char *)&dim, sizeof(int));

  char projected_data_short_name[16] = "";
  snprintf(projected_data_short_name, 16, "%s", projected_parameter.c_str());
  // sprintf(projected_data_short_name, "%16c", projected_parameter.c_str());

  binary_file.write((char *)projected_data_short_name, sizeof(char) * 16);

  for (unsigned int idim = 0; idim < dim; ++idim) {

    char axis_short_name[8] = "";
    snprintf(axis_short_name, 8, "%s", axis[idim].c_str());

    binary_file.write((char *)&axis_short_name, sizeof(char) * 8);
    binary_file.write((char *)&min[idim], sizeof(double));
    binary_file.write((char *)&max[idim], sizeof(double));
    binary_file.write((char *)&n_cells[idim], sizeof(std::size_t));

    data_size *= n_cells[idim];
  }

  binary_file.write((char *)data, sizeof(double) * data_size);

  binary_file.close();
}

// _____________________________________________________________________
//! \brief output a structured array of any dimension in a binary file

//! WARNING: This function is not optimized
//! This function should be put in subdomain or in a specific diag class

//! \param[in] file_name - name of the file on disk
//! \param[in] projected_parameter - name of the data parameter
//! \param[in] axis - axis names
//! \param[in] min - min value for each axis
//! \param[in] max - max value for each axis
// _____________________________________________________________________
void output_vtk_structured_grid(const std::string& file_name,
                                const std::string& projected_parameter,
                                const std::vector<std::string>& axis,
                                const std::vector<std::size_t>& n_cells,
                                const std::vector<double>& min,
                                const std::vector<double>& delta,
                                double *__restrict__ data) {

  const unsigned int dim = axis.size();
  unsigned int data_size = 1;
  for (unsigned int idim = 0; idim < dim; ++idim) {
    data_size *= n_cells[idim];
  }

  std::ofstream binary_file(file_name, std::ios::out);

  binary_file << "# vtk DataFile Version 2.0\n"
              << projected_parameter.c_str() << "\n"
              << "ASCII\n";
  binary_file << "DATASET STRUCTURED_POINTS\n";

  // Add the dimensions
  binary_file << "DIMENSIONS ";
  for (int idim = dim - 1; idim >= 0; --idim) {
    binary_file << n_cells[idim] << " ";
  }
  binary_file << "\n";

  // Add the aspect ratio
  binary_file << "SPACING ";
  for (int idim = dim - 1; idim >= 0; --idim) {
    binary_file << delta[idim] << " ";
  }
  binary_file << "\n";

  // Origin
  binary_file << "ORIGIN ";
  for (int idim = dim - 1; idim >= 0; --idim) {
    binary_file << min[idim] << " ";
  }
  binary_file << "\n";

  // data
  binary_file << "POINT_DATA " << data_size << "\n"
              << "SCALARS " << projected_parameter.c_str() << " double 1\n"
              << "LOOKUP_TABLE default\n";

  for (std::size_t i = 0; i < data_size; ++i) {
    binary_file << data[i] << " ";
  }
  binary_file.close();
}

// _____________________________________________________________________
//! \brief Particle binning diagnostic for any dimension (1d, 2d, 3d, and more)

//! NOTE: if for a given axis, max <= min, then the min and max values are automatically computed

//! WARNING: This function is not optimized
//! This function should be put in subdomain or in a specific diag class

//! \param[in] diag_name - string used to initiate the diag name
//! \param[in] patches - vector of patches
//! \param[in] projected_parameter - property projected on the grid, can be
//! `weight`
//! \param[in] axis - axis to use for the grid. The axis vector size
//! determines the dimension of the diag. Axis can be `gamma`, `weight`, `x`,
//! `y`, `z`, `px`, `py`, `pz`
//! \param[in] n_cells - number of cells for each axis
//! \param[in] min - min value for each axis
//! \param[in] max - max value for each axis
//! \param[in] is - species
//! \param[in] it - iteration
//! \param[in] format - (optional argument) - determine the output format, can
//! be `binary` (default) or `vtk`
// _____________________________________________________________________
void particle_binning(std::string& diag_name,
                      Params &params,
                      Particles &particles,
                      std::string projected_parameter,
                      const std::vector<std::string>& axis,
                      const std::vector<std::size_t>& n_cells,
                      const std::vector<double>& min_in,
                      const std::vector<double>& max_in,
                      int is,
                      int it,
                      std::string format = "binary",
                      bool verbose       = false) {

  if (verbose) {
    std::cout << " create particle binning: " << diag_name << std::endl;
    std::cout << " - it: " << it << std::endl;
  }

  // Get diag dimension
  const unsigned int dim = axis.size();

  // compute the total size of the diag grid
  std::size_t diag_data_size = n_cells[0];
  for (unsigned int idim = 1; idim < dim; ++idim) {
    diag_data_size *= n_cells[idim];
  }

  // Allocate diag grid
  std::vector<double> diag_data(diag_data_size, 0.);

  // Use an int code for the axis instead of string
  int axis_code[3];
  for (unsigned int idim = 0; idim < dim; ++idim) {

    if (axis[idim] == "gamma") {
      axis_code[idim] = 0;
    } else if (axis[idim] == "x") {
      axis_code[idim] = 1;
    } else if (axis[idim] == "y") {
      axis_code[idim] = 2;
    } else if (axis[idim] == "z") {
      axis_code[idim] = 3;
    } else if (axis[idim] == "px") {
      axis_code[idim] = 4;
    } else if (axis[idim] == "py") {
      axis_code[idim] = 5;
    } else if (axis[idim] == "pz") {
      axis_code[idim] = 6;
    }
  }

  // Use a code for the projected parameter
  int data_code = -1;
  if (projected_parameter == "weight") {
    data_code = 0;
  } else if (projected_parameter == "density") {
    data_code = 1;
  } else if (projected_parameter == "particles") {
    data_code = 2;
  }

  // Check the min max values
  double min_value[3];
  double max_value[3];

  for (unsigned int idim = 0; idim < dim; ++idim) {

    min_value[idim] = min_in[idim];
    max_value[idim] = max_in[idim];

    // We compute the boundaries in this case
    // if min_in <= max_in
    if (min_value[idim] >= max_value[idim]) {

      if (axis_code[idim] == 0) {
        min_value[idim] = 1e9;
        max_value[idim] = -1e9;
      }

        const std::size_t n_particles = particles.size();
        for (std::size_t ip = 0; ip < n_particles; ip++) {

          double value = 0.0;

          if (axis_code[idim] == 0) {
            value = sqrt(1 +
                         particles.mx_h_m(ip) *
                           particles.mx_h_m(ip) +
                         particles.my_h_m(ip) *
                           particles.my_h_m(ip) +
                         particles.mz_h_m(ip) *
                           particles.mz_h_m(ip));
          } else if (axis_code[idim] == 1) {
            value = particles.x_h_m(ip);
          } else if (axis_code[idim] == 2) {
            value = particles.y_h_m(ip);
          } else if (axis_code[idim] == 3) {
            value = particles.z_h_m(ip);
          } else if (axis_code[idim] == 4) {
            value = particles.mx_h_m(ip);
          } else if (axis_code[idim] == 5) {
            value = particles.my_h_m(ip);
          } else if (axis_code[idim] == 6) {
            value = particles.mz_h_m(ip);
          }

          min_value[idim] = std::min(min_value[idim], static_cast<double>(value));
          max_value[idim] = std::max(max_value[idim], static_cast<double>(value));
        } // end for each particles

      max_value[idim] *= 1.01;

      if (verbose) {
        std::cout << " - Auto min/max for axis " << axis[idim] << " " << min_value[idim] << " "
                  << max_value[idim] << std::endl;
      }
    }
  }

  // Delta
  std::vector<double> delta(dim);
  for (unsigned int idim = 0; idim < dim; ++idim) {
    delta[idim] = (max_value[idim] - min_value[idim]) / n_cells[idim];
  }


    // get number of particles to project
    const std::size_t n_particles = particles.size();

    // Compute data
    for (std::size_t ip = 0; ip < n_particles; ip++) {

      double value[3];
      bool inside_diag_data = true;

      for (unsigned int idim = 0; idim < dim; ++idim) {

        if (axis_code[idim] == 0) {
          value[idim] = sqrt(
            1 +
            particles.mx_h_m(ip) * particles.mx_h_m(ip) +
            particles.my_h_m(ip) * particles.my_h_m(ip) +
            particles.mz_h_m(ip) * particles.mz_h_m(ip));
        } else if (axis_code[idim] == 1) {
          value[idim] = particles.x_h_m(ip);
        } else if (axis_code[idim] == 2) {
          value[idim] = particles.y_h_m(ip);
        } else if (axis_code[idim] == 3) {
          value[idim] = particles.z_h_m(ip);
        } else if (axis_code[idim] == 4) {
          value[idim] = particles.mx_h_m(ip);
        } else if (axis_code[idim] == 5) {
          value[idim] = particles.my_h_m(ip);
        } else if (axis_code[idim] == 6) {
          value[idim] = particles.mz_h_m(ip);
        }

        if (value[idim] < min_value[idim] || value[idim] >= max_value[idim]) {
          // Compute global cell index
          inside_diag_data = false;
        }
      }

      if (inside_diag_data) {

        int index[3];
        unsigned int global_index = 0;

        // For each axis, we compute the value to be projected and the grid
        // index
        for (unsigned int idim = 0; idim < dim; ++idim) {

          // Compute global cell index
          index[idim] = static_cast<int>(floor((value[idim] - min_value[idim]) / delta[idim]));

          // Compute the global 1d index in diag_data
          global_index = global_index * n_cells[idim] + index[idim];
        }

        if (data_code == 0) {
          diag_data[global_index] += static_cast<double>(particles.weight_h_m(ip));
        } else if (data_code == 1) {
          diag_data[global_index] +=
            static_cast<double>(particles.weight_h_m(ip) * params.inv_cell_volume);
        } else if (data_code == 2) {
          diag_data[global_index] += 1;
        }
      }
    } // end for each particles


  // _______________________________________________________________
  // Save the diag_data

  // Output name template
  std::string template_string = "%s_s%02d_%0" + std::to_string(params.max_it_digits) + "d";

  if (format == "binary" or format == "bin") {
    template_string += ".bin";
  } else if (format == "vtk") {
    template_string += ".vtk";
  }

  char buffer[64];
  snprintf(buffer, 64, template_string.c_str(), diag_name.c_str(), is, it);

  std::string file_name(params.output_directory + "/" + buffer);

  // Binary format
  if (format == "binary" or format == "bin") {

    Diags::output_binary_structured_grid(file_name,
                                         projected_parameter,
                                         axis,
                                         n_cells,
                                         {min_value[0], min_value[1], min_value[2]},
                                         {max_value[0], max_value[1], max_value[2]},
                                         diag_data.data());

    // Old style VTK format
  } else if (format == "vtk") {

    output_vtk_structured_grid(file_name,
                               projected_parameter,
                               axis,
                               n_cells,
                               {min_value[0], min_value[1], min_value[2]},
                               {delta[0], delta[1], delta[2]},
                               diag_data.data());
  }
}

// _____________________________________________________________________
//! \brief output the field grids

//! NOTE: Create a binary file per field component

//! \param[in] Params - global constant parameters
//! \param[in] em - species
//! \param[in] it - iteration
//! \param[in] format - (optional argument) - determine the output format, can
//! be `binary` (default) or `vtk`
// _____________________________________________________________________
void fields(Params &params, ElectroMagn &em, unsigned int it, std::string format = "binary") {

  // Output name template
  std::string template_string = "%s_%0" + std::to_string(params.max_it_digits) + "d";

  if (format == "binary" or format == "bin") {
    template_string += ".bin";
  } else if (format == "vtk") {
    template_string += ".vtk";
  }

  char buffer[64];

  std::vector<ElectroMagn::hostview_t *> field_list =
    {&em.Ex_h_m, &em.Ey_h_m, &em.Ez_h_m, &em.Bx_h_m, &em.By_h_m, &em.Bz_h_m};

  // Needed because the mirrored view have "_mirror" appended to their name
  std::vector<ElectroMagn::view_t *> device_field_list =
    {&em.Ex_m, &em.Ey_m, &em.Ez_m, &em.Bx_m, &em.By_m, &em.Bz_m};


  for (size_t ifield = 0; ifield < field_list.size(); ++ifield) {

    const int nx = field_list[ifield]->extent(0);
    const int ny = field_list[ifield]->extent(1);
    const int nz = field_list[ifield]->extent(2);

    const char* field_name = device_field_list[ifield]->label().c_str();

    double *tmp_grid = new double[field_list[ifield]->size()];

    for (int ix = 0; ix < nx; ++ix) {
      for (int iy = 0; iy < ny; ++iy) {
        for (int iz = 0; iz < nz; ++iz) {
          int i       = ix * ny * nz + iy * nz + iz;
          tmp_grid[i] = static_cast<double>((*field_list[ifield])(ix, iy, iz));
        }
      }
    }

    // Binary format
    if (format == "binary" or format == "bin") {
      snprintf(buffer, 64, template_string.c_str(), field_name, it);
      std::string file_name(params.output_directory + "/" + buffer);

      output_binary_structured_grid(
        file_name,
        field_name,
        {"x", "y", "z"},
        {field_list[ifield]->extent(0), field_list[ifield]->extent(1), field_list[ifield]->extent(2)},
        {params.inf_x, params.inf_y, params.inf_z},
        {params.sup_x, params.sup_y, params.sup_z},
        tmp_grid);

      // Old style vtk format
    } else if (format == "vtk") {

      snprintf(buffer, 64, template_string.c_str(), field_name, it);
      std::string file_name(params.output_directory + "/" + buffer);

      output_vtk_structured_grid(
        file_name,
        field_name,
        {"x", "y", "z"},
        {field_list[ifield]->extent(0), field_list[ifield]->extent(1), field_list[ifield]->extent(2)},
        {params.inf_x, params.inf_y, params.inf_z},
        {params.dx, params.dy, params.dz},
        tmp_grid);
    }

    delete[] tmp_grid;
  }
}

// _____________________________________________________________________
//! \brief Particle cloud diagnostic
//!        Dump all particle properties in a binary or vtk file.

//! NOTE: This function should be put in subdomain or in a specific diag class

//! \param[in] diag_name - string used to initiate the diag name
//! \param[in] patches - vector of patches
//! \param[in] is - species
//! \param[in] it - iteration
//! \param[in] format - output format, can be "binary" (default) or "vtk". This
//! defines the file extension, respectively `.bin` or `.vtk`
// _____________________________________________________________________
void particle_cloud(const std::string& diag_name,
                    Params &params,
                    Particles &particles,
                    int is,
                    int it,
                    const std::string& format = "binary") {

  unsigned int number_of_particles = particles.size();

  // Output name template
  // binary: %s_s%02d_%0<max_it_digits>d.bin
  // vtk:    %s_s%02d_%0<max_it_digits>d.vtk
  // where %s is the diag_name, %02d is the species index
  // and %0<max_it_digits>d the iteration number
  std::string template_string = "%s_s%02d_%0" + std::to_string(params.max_it_digits) + "d";

  if (format == "binary" or format == "bin") {

    template_string += ".bin";

    char buffer[64];
    snprintf(buffer, 64, template_string.c_str(), diag_name.c_str(), is, it);

    std::string file_name(params.output_directory + "/" + buffer);

    std::ofstream binary_file(file_name, std::ios::out | std::ios::binary);

    binary_file.write((char *)&number_of_particles, sizeof(number_of_particles));

    if (!binary_file) {
      ERROR(" Error while creating the file :" << file_name)
      std::raise(SIGABRT);
    }


      const std::size_t n_particles = particles.size();
      for (std::size_t ip = 0; ip < n_particles; ++ip) {

        binary_file.write((char *)(&particles.weight_h_m(ip)), sizeof(double));

        binary_file.write((char *)(&particles.x_h_m(ip)), sizeof(double));
        binary_file.write((char *)(&particles.y_h_m(ip)), sizeof(double));
        binary_file.write((char *)(&particles.z_h_m(ip)), sizeof(double));

        binary_file.write((char *)(&particles.mx_h_m(ip)), sizeof(double));
        binary_file.write((char *)(&particles.my_h_m(ip)), sizeof(double));
        binary_file.write((char *)(&particles.mz_h_m(ip)), sizeof(double));

      } // End particle loop

    // Cleaning
    binary_file.close();

  } else if (format == "vtk") {

    template_string += ".vtk";

    char buffer[64];
    snprintf(buffer, 64, "%s_s%02d_%05d.vtk", diag_name.c_str(), is, it);

    std::string file_name(params.output_directory + "/" + buffer);

    std::ofstream vtk_file(file_name, std::ios::out | std::ios::trunc);

    if (!vtk_file) {
      std::cerr << " Error while creating the file :" << file_name << std::endl;
      std::raise(SIGABRT);
    }

    vtk_file << "# vtk DataFile Version 3.0" << std::endl;
    vtk_file << "vtk output" << std::endl;
    vtk_file << "ASCII" << std::endl;
    vtk_file << "DATASET POLYDATA" << std::endl;

    // Particle positions
    vtk_file << std::endl;
    vtk_file << "POINTS " << number_of_particles << " float" << std::endl;

      const std::size_t n_particles = particles.size();
      for (std::size_t ip = 0; ip < n_particles; ++ip) {

        vtk_file << particles.z_h_m(ip) << " "
                 << particles.y_h_m(ip) << " "
                 << particles.x_h_m(ip) << std::endl;

      } // End particle loop

    // Construction of the weight
    vtk_file << std::endl;
    vtk_file << "POINT_DATA " << number_of_particles << std::endl;
    vtk_file << "SCALARS weight float" << std::endl;
    vtk_file << "LOOKUP_TABLE default" << std::endl;

      for (std::size_t ip = 0; ip < particles.size(); ++ip) {

        vtk_file << particles.weight_h_m(ip) << " ";

      } // End particle loop

    vtk_file << std::endl;

    // Construction of the energy
    vtk_file << std::endl;
    vtk_file << "SCALARS gamma float" << std::endl;
    vtk_file << "LOOKUP_TABLE default" << std::endl;


      for (std::size_t ip = 0; ip < particles.size(); ++ip) {

        const double gamma =
          1 /
          sqrt(
            1. +
            particles.mx_h_m(ip) * particles.mx_h_m(ip) +
            particles.my_h_m(ip) * particles.my_h_m(ip) +
            particles.mz_h_m(ip) * particles.mz_h_m(ip));

        vtk_file << gamma << " ";
      }

    vtk_file << std::endl;

    // Construction of the momentum vector
    vtk_file << std::endl;
    vtk_file << "VECTORS momentum float" << std::endl;
      for (std::size_t ip = 0; ip < particles.size(); ++ip) {
        vtk_file << particles.mx_h_m(ip) << " ";
        vtk_file << particles.my_h_m(ip) << " ";
        vtk_file << particles.mz_h_m(ip) << " ";
      }

    vtk_file.close();
  }
}

// _____________________________________________________________________
//! \brief Particle scalars
//!        Compute reduced 0D scalars for particles

//! NOTE: Create an ascii file called `species_*.txt` where `*` is the species
//! number `is`.

//! NOTE: If the parameter `it` is equal to 0, the file is reinitialized
//! (ios::trunc). Else, the new data is added at the end of the file.

//! NOTE: The first line of the file is a header describing all columns.

//! \param[in] Params - global constant parameters
//! \param[in] patches - vector of patches
//! \param[in] is - species
//! \param[in] it - iteration
// _____________________________________________________________________
void scalars(Params &params, Particles &particles, unsigned int is, unsigned int it) {

  // Particle scalars _________________________________________

  unsigned int number_of_particles  = 0;
  double species_kinetic_energy = 0;

    species_kinetic_energy += particles.get_kinetic_energy(minipic::device);
    number_of_particles += particles.size();

  // output scalars file _____________________________________

  std::ofstream scalars_file;

  char buffer[64];
  snprintf(buffer, 64, "species_%02d.txt", is);

  std::string file_name(params.output_directory + "/" + buffer);

  if (it == 0) {
    scalars_file.open(file_name, std::ios::out | std::ios::trunc);
  } else {
    scalars_file.open(file_name, std::ios::out | std::ios::app);
  }

  if (!scalars_file.is_open()) {
    std::cerr << " Error while accessing the scalars file." << std::endl;
    std::raise(SIGABRT);
  }

  // Header
  if (it == 0) {
    scalars_file << "iter number_of_particles E_kin \n";
  }

  // Data
  scalars_file << it;
  scalars_file << " " << number_of_particles;
  scalars_file << " " << std::scientific << std::setprecision(15) << species_kinetic_energy;
  scalars_file << "\n";

  scalars_file.close();
}

// _____________________________________________________________________
//! \brief Field scalars
//!        Compute reduced 0D scalars for Fields

//! NOTE: Create an ascii file called `fields.txt`.

//! NOTE: If the parameter `it` is equal to 0, the file is reinitialized
//! (ios::trunc). Else, the new data is added at the end of the file.

//! NOTE: The first line of the file is a header describing all columns.

//! \param[in] Params - global constant parameters
//! \param[in] em - electromagnetic object containing all fields
//! \param[in] it - iteration
// _____________________________________________________________________
void scalars(Params &params, ElectroMagn &em, unsigned int it) {

  // compute Field scalars _________________________________________

  double Ex_energy = 0.5 * operators::sum_power(em.Ex_m, 2) * params.cell_volume;
  double Ey_energy = 0.5 * operators::sum_power(em.Ey_m, 2) * params.cell_volume;
  double Ez_energy = 0.5 * operators::sum_power(em.Ez_m, 2) * params.cell_volume;

  double Bx_energy = 0.5 * operators::sum_power(em.Bx_m, 2) * params.cell_volume;
  double By_energy = 0.5 * operators::sum_power(em.By_m, 2) * params.cell_volume;
  double Bz_energy = 0.5 * operators::sum_power(em.Bz_m, 2) * params.cell_volume;

  double Jx_energy = 0.5 * operators::sum_power(em.Jx_m, 2) * params.cell_volume;
  double Jy_energy = 0.5 * operators::sum_power(em.Jy_m, 2) * params.cell_volume;
  double Jz_energy = 0.5 * operators::sum_power(em.Jz_m, 2) * params.cell_volume;

  // #endif

  // output Field scalars _________________________________________

  std::ofstream scalars_file;

  std::string file_name(params.output_directory + "/fields.txt");

  if (it == 0) {
    scalars_file.open(file_name, std::ios::out | std::ios::trunc);
  } else {
    scalars_file.open(file_name, std::ios::out | std::ios::app);
  }

  if (!scalars_file.is_open()) {
    std::cerr << " Error while accessing the file `fields.txt`." << std::endl;
    std::raise(SIGABRT);
  }

  // Header
  if (it == 0) {
    scalars_file << "dt: " << params.dt << "\n";
    scalars_file << "iter Ex Ey Ez Bx By Bz Jx Jy Jz\n";
  }

  // Data
  scalars_file << it;
  scalars_file << " " << std::scientific << std::setprecision(15) << Ex_energy;
  scalars_file << " " << std::scientific << std::setprecision(15) << Ey_energy;
  scalars_file << " " << std::scientific << std::setprecision(15) << Ez_energy;

  scalars_file << " " << std::scientific << std::setprecision(15) << Bx_energy;
  scalars_file << " " << std::scientific << std::setprecision(15) << By_energy;
  scalars_file << " " << std::scientific << std::setprecision(15) << Bz_energy;

  scalars_file << " " << std::scientific << std::setprecision(15) << Jx_energy;
  scalars_file << " " << std::scientific << std::setprecision(15) << Jy_energy;
  scalars_file << " " << std::scientific << std::setprecision(15) << Jz_energy;
  scalars_file << "\n";

  scalars_file.close();
}

} // namespace Diags

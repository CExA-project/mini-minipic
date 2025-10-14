/* _____________________________________________________________________ */
//! \file Main.cpp

//  _ __ ___ (_)_ __ (_)_ __ (_) ___
// | '_ ` _ \| | '_ \| | '_ \| |/ __|
// | | | | | | | | | | | |_) | | (__
// |_| |_| |_|_|_| |_|_| .__/|_|\\___|
//                     |_|

//! \brief Main file for Minipic
//! NOTE: this program is intended for computer science studies
//! and should be not used for physics simulation
/* _____________________________________________________________________ */

#include "Params.hpp"
#include "SubDomain.hpp"

// Load a setup
#include "study_thermal.hpp"
// #include "study_sorted.hpp"
// #include "study_random.hpp"
// #include"default_gpu.hpp"

//! Main function
int main(int argc, char *argv[]) {

  // ___________________________________________
  // Setup input parameters in struct

  // Create the gloal parameters
  Params params;

  // Print the Minipic title
  params.title();

  // default parameters
  setup(params);

  // change from command line arguments
  params.read_from_command_line_arguments(argc, argv);

  // Initialize the main parameters for the simulation
  params.compute();

  // Print a summary of input parameters
  params.info();

  // Initialize Kokkos
  Kokkos::initialize(argc, argv);

  // Print the backend information
  Kokkos::print_configuration(std::cout);

  {

    // Timers initialization
    Timers timers(params);

    // ______________________________________________________
    //
    // Initialization
    // ______________________________________________________

    SubDomain subdomain;

    // Creation of the domain
    subdomain.allocate(params);

    // Initialization of the diagnostics
    Diags::initialize(params);

    timers.stop(timers.initialization);
    timers.save_initialization();
    timers.start(timers.main_loop);

    // ______________________________________________________
    //
    // Initial diagnostics
    // ______________________________________________________

    timers.start(timers.diags);
    subdomain.diagnostics(params, 0);
    timers.stop(timers.diags);

    // ______________________________________________________
    //
    // main PIC loop
    // ______________________________________________________

    printf("\n #########    START COMPUTE    ############\n");
    std::cout << " -----------------------------------------------------|" << std::endl;
    std::cout << "           |       |           | Elapsed  | Remaining |" << std::endl;
    std::cout << " Iteration |    %  | Particles | Time (s) | Time (s)  |" << std::endl;
    std::cout << " ----------|-------|-----------|----------|-----------|" << std::endl;

    timers.start(timers.main_loop);

    DEBUG("Start of main loop");

    // start main loop
    for (unsigned int it = 1; it <= params.n_it; it++) {

      timers.start(timers.pic_iteration);

      // Single PIC iteration
      subdomain.iterate(params, it);

      timers.stop(timers.pic_iteration);

      timers.start(timers.diags);

      // Diagnostics
      subdomain.diagnostics(params, it);

    if (!(it % params.print_period)) {

        const unsigned int total_number_of_particles =
          subdomain.get_total_number_of_particles();

        double elapsed_time   = timers.get_elapsed_time();
        double remaining_time = elapsed_time / it * (params.n_it - it);

        std::cout << " " << std::setw(9) << it;
        std::cout << " | " << std::fixed << std::setprecision(1) << std::setw(5)
                  << static_cast<float>(it) / static_cast<float>(params.n_it) * 100;
        std::cout << " | " << std::scientific << std::setprecision(2) << std::setw(9)
                  << total_number_of_particles;
        std::cout << " | " << std::scientific << std::setprecision(2) << std::setw(8)
                  << elapsed_time;
        std::cout << " | " << std::scientific << std::setprecision(2) << std::setw(9)
                  << remaining_time;
        std::cout << " | " << std::endl;
      }
      timers.stop(timers.diags);

      timers.save(params, it);

    } // end main loop

    DEBUG("End of main loop");

    timers.stop(timers.main_loop);

    std::cout << "#########    END COMPUTE    ############\n\n";
    printf("\n");

    // ____________________________________________________
    // Print timers

    timers.print();

    timers.save(params, params.n_it + 1);

  }

  Kokkos::finalize();

  std::cerr << "> minipic finalized" << std::endl;
  return 0;
}

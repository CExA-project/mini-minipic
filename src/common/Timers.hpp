/* _____________________________________________________________________ */
//! \file Timers.cpp

//! \brief Timer class to measure the time spent in each part of the code

/* _____________________________________________________________________ */

#pragma once

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/time.h>
#include <time.h>
#include <vector>

#include "Params.hpp"


// _______________________________________________________________
//
//! Structure representing a code section
// _______________________________________________________________
struct Section {
  int id;
  std::string name;
};

// _______________________________________________________________
//
//! \brief Timer class to measure the time spent in each part of the code
//! \details each timer contains N_patches + 1 counters :
//! - the first one is the global counter
//! - the others are the thread counters (one per patch)
// _______________________________________________________________
template <class T_clock> class Struc_Timers {
public:
  // List of available code sections
  const Section initialization = Section{0, "initialization"};
  const Section main_loop      = Section{1, "main loop"};
  const Section pic_iteration  = Section{2, "pic_iteration"};
  const Section diags          = Section{3, "all diags"};

  // Vector of timers
  std::vector<Section> sections = {initialization,
                                   main_loop,
                                   pic_iteration,
                                   diags};

  // String Buffer to store the timers
  std::stringstream timers_buffer;

  // _______________________________________________________________
  //
  //! Constructor
  // _______________________________________________________________
  Struc_Timers(Params params) {
    // By default there are 2 timers :
    // - initilization
    // - main loop

    temporary_times.resize(sections.size());
    accumulated_times.resize(sections.size());

    // initialize timers
    for (std::size_t i = 0; i < sections.size() ; i++) {
      accumulated_times[i] = 0;
      temporary_times[i]   = T_clock::now();
    }

    // Create a new timers.json file
    std::ofstream file;
    file.open("timers.json");

    // Add parameters
    file << "{\n";
    file << "  \"parameters\" : {\n";
    file << "    \"iterations\" : " << params.n_it << ",\n";
    file << "    \"save_timers_period\" : " << params.save_timers_period << ",\n";
    file << "    \"save_timers_start\" : " << params.save_timers_start << "\n";
    file << "  },\n";

    // close
    file.close();
  }

  //! Destructor
  ~Struc_Timers() {};

  // _______________________________________________________________
  //
  //! Start a global timer
  // _______________________________________________________________
  void start(Section section) {

    // auto time = std::chrono::high_resolution_clock::now();

    auto index = first_index(section);
    temporary_times[index] = T_clock::now();

    // std::cout << "Start timer " << section.name << " at " <<
    // temporary_times[index].time_since_epoch().count() << std::endl;
  }

  // _______________________________________________________________
  //
  //! Stop a timer
  // _______________________________________________________________
  void stop(Section section) {

    auto index = first_index(section);

    // auto time = std::chrono::high_resolution_clock::now();
    auto time = T_clock::now();

    std::chrono::duration<double> diff = time - temporary_times[index];
    accumulated_times[index] += diff.count();

    // std::cout << "Stop timer " << section.name << " at " << time.time_since_epoch().count()
    //           << " with diff " << diff.count()
    //           << std::endl;
  }

  // _______________________________________________________________
  //
  //! Get the elapsed time since the beginning of the time loop
  // _______________________________________________________________
  double get_elapsed_time() {
    std::chrono::duration<double> diff =
      // std::chrono::high_resolution_clock::now() - temporary_times[1];
      T_clock::now() - temporary_times[1];
    return diff.count();
  }

  // _______________________________________________________________
  //
  //! \brief Print all timers
  // _______________________________________________________________
  void print() {
    double percentage;

    const double initialization_time = accumulated_times[first_index(initialization)];
    const double main_loop_time      = accumulated_times[first_index(main_loop)];
    const double diags_time          = accumulated_times[first_index(diags)];
    const double pic_iteration_time  = accumulated_times[first_index(pic_iteration)];

    double total_time = initialization_time + main_loop_time;

    printf(" ---------------------------------------------- |\n");
    printf(" Global timers                                  |\n");
    printf(" ---------------------------------------------- |\n");
    printf("            code part |  time (s)  | percentage |\n");
    printf(" ---------------------|------------|----------- |\n");

    percentage = initialization_time / total_time * 100;
    printf("%21s |%11.6lf |%9.2lf %% |\n",
           initialization.name.c_str(),
           initialization_time,
           percentage);

    percentage = main_loop_time / total_time * 100;
    printf("%21s |%11.6lf |%9.2lf %% |\n", main_loop.name.c_str(), main_loop_time, percentage);

    printf(" ---------------------------------------------- |\n");
    printf(" Main loop                                      |\n");
    printf(" ---------------------------------------------- |\n");
    printf("            code part |  time (s)  | percentage |\n");
    printf(" ---------------------|------------|----------- |\n");

    total_time = pic_iteration_time + diags_time;
    percentage = pic_iteration_time / total_time * 100;
    const std::string pic_iterations_name = "PIC iterations";
    printf("%21s |%11.6lf |%9.2lf %% |\n",
           pic_iterations_name.c_str(),
           pic_iteration_time,
           percentage);

    percentage = diags_time / total_time * 100;
    const std::string diags_name = "Diagnostics";
    printf("%21s |%11.6lf |%9.2lf %% |\n", diags_name.c_str(), diags_time, percentage);

  }

  // _______________________________________________________________
  //
  //! \brief Save the initialization time in the timers file
  //
  // _______________________________________________________________
  void save_initialization() {
    const double initialization_time = accumulated_times[first_index(initialization)];
    std::ofstream file;
    file.open("timers.json", std::ios::app);
    file << "  \"initialization\" : " << initialization_time << ",\n";
    file.close();
  }

  // _______________________________________________________________
  //
  //! \brief  Write all timers in a file using json format
  //
  //! \details The file is named "timers.json" and has the following format :
  //! {
  //!   "initilization" : [global],
  //!   "0" : {
  //!       "pic iterations" : [global],
  //!       "diags" : [global],
  //!       "interpolate" : [global],
  //!       "push" : [global],
  //!     ....
  //!   },
  //!   "10" : {
  //!     "pic iterations" : [global],
  //!     "diags" : [global],
  //!     "interpolate" : [global],
  //!     "push" : [global],
  //!     ....
  //!   }
  //!   "final" : {
  //!     "pic iterations" : [global],
  //!     "diags" : [global],
  //!     "interpolate" : [global],
  //!     "push" : [global],
  //!     ....
  //!   }
  //!   "main loop" : [global]
  //! ...
  //! }
  //! Use the scientific format with 6 digits after the comma
  //
  //! \param  params : parameters of the simulation
  //! \param  iteration : current iteration
  // _______________________________________________________________
  void save(Params params, unsigned int iteration) {

    if (iteration < params.save_timers_start)
      return;

    // iteration since the start of the saving
    unsigned int timer_iteration = iteration - params.save_timers_start;

    // save the timers every save_timers_period iterations
    // or if the simulation is finished (iteration > n_it)
    if (!(timer_iteration % params.save_timers_period) || (iteration > params.n_it)) {

      // const double initialization_time = accumulated_times[first_index(initialization)];
      const double main_loop_time   = accumulated_times[first_index(main_loop)];
      const double diags_time       = accumulated_times[first_index(diags)];
      const auto pic_iteration_time = accumulated_times[first_index(pic_iteration)];

      std::stringstream local_buffer("");

      local_buffer << std::scientific;
      if (iteration <= params.n_it) {
        local_buffer << "  \"" << iteration << "\" : {\n";
      } else {
        local_buffer << "  \"final\" : {\n";
      }
      local_buffer << "    \"pic iteration\" : " << pic_iteration_time << ",\n";
      local_buffer << "    \"diags\" : " << diags_time << ",\n";

      for (std::size_t itimer = 3; itimer < sections.size(); itimer++) {
        local_buffer << "    \"" << sections[itimer].name << "\" : [";
          local_buffer << accumulated_times[first_index(sections[itimer])];
        if (itimer < sections.size() - 1) {
          local_buffer << "],\n";
        } else {
          local_buffer << "]\n";
        }
      }
      local_buffer << "  },\n";

      if (iteration > params.n_it) {
        local_buffer << "  \"main loop\" : " << main_loop_time << "\n";
        local_buffer << "}\n";
      }

      // std::cout << " -> Save timers at iteration " << iteration << std::endl;

      if (params.bufferize_timers_output) {

        timers_buffer << local_buffer.str();

        if (iteration > params.n_it) {
          std::ofstream file;
          file.open("timers.json", std::ios::app);
          file << timers_buffer.str();
          file.close();
        }

      } else {
        std::ofstream file;
        file.open("timers.json", std::ios::app);
        file << local_buffer.str();
        file.close();
      }

    } // end if save_timers_period
  } // end save

private:
  // Array to store the timers
  std::vector<double> accumulated_times;

  // Array to store temporary values
  std::vector<std::chrono::time_point<T_clock>> temporary_times;

  // Number of patches
  int N_patches;

  //! Get first index from section id
  int first_index(Section section) { return section.id ; }

  //! Select a steady clock
  // static constexpr auto clock() {
  //   if constexpr (std::chrono::high_resolution_clock::is_steady) {
  //     return std::chrono::high_resolution_clock();
  //   } else {
  //     return std::chrono::steady_clock();
  //   }
  // }
};

// Timers shortcut
// using Timers = Struc_Timers<std::chrono::high_resolution_clock>;
using Timers = Struc_Timers<std::chrono::steady_clock>;

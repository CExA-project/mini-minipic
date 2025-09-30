/* _____________________________________________________________________ */
//! \file Backend.hpp

//! \brief determine the best backend to use

/* _____________________________________________________________________ */

#ifndef BACKEND_H
#define BACKEND_H

#include "Params.hpp"

// _____________________________________________________________________
//
// Backend class
//
//! \brief manage the backend properties
// _____________________________________________________________________

class Backend {
public:
  // _____________________________________________________________________
  // Public parameters

  int number_of_threads;

  // _____________________________________________________________________
  // Public methods

  Backend() {}

  ~Backend() {}

  // _____________________________________________________________________
  //
  //! \brief Initialize the backend
  //! \param argc number of arguments
  //! \param argv arguments
  //! \param params global parameters
  // _____________________________________________________________________
  void init([[maybe_unused]] int argc,
            [[maybe_unused]] char *argv[],
            [[maybe_unused]] const Params &params) {

#if defined(__MINIPIC_OMP__)
    number_of_threads = omp_get_max_threads();
#else
    number_of_threads = 1;
#endif


#if defined(__MINIPIC_KOKKOS_COMMON__)

    Kokkos::initialize(argc, argv);

#if defined(__MINIPIC_KOKKOS_UNIFIED__)
    static_assert(Kokkos::has_shared_space, "code only works on backends with SharedSpace");
#endif
#endif

  }

  // _____________________________________________________________________
  //
  //! \brief Finalize the backend
  // _____________________________________________________________________
  void finalize() {
#if defined(__MINIPIC_KOKKOS_COMMON__)
    Kokkos::finalize();
#endif
  }

  // _____________________________________________________________________
  //
  //! \brief Print the backend information
  // _____________________________________________________________________
  void info() {
    std::cout << " > Backend: " << std::endl;
#ifdef BACKEND
    std::cout << "   - CMake Name: " << BACKEND << std::endl;
#endif

#if defined(__MINIPIC_OMP__)
    std::cout << "   - Selected parallel programming model: OpenMP for" << std::endl;
    std::cout << "   - OMP number of threads: " << number_of_threads << std::endl;
#endif

#if defined(__MINIPIC_KOKKOS_COMMON__)
    std::cout << "   - Selected parallel programming model: Kokkos" << std::endl;
    std::cout << "   - Device execution Space: " << typeid(Kokkos::DefaultExecutionSpace).name()
              << std::endl;
    std::cout << "   - Host execution Space: " << typeid(Kokkos::DefaultHostExecutionSpace).name()
              << std::endl;
    // std::cout << "   - Number of threads: " << &Kokkos::num_threads << std::endl;
    Kokkos::print_configuration(std::cout);
#endif

  }
};

#endif

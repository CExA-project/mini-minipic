# Developer zone

## Domain decomposition

(mini)miniPIC does not support distributed memory parallelism and contains a single domain.

## PIC loop steps

<img title="pic loop" alt="pic loop" src="./images/pic_loop.png" />

## Code design

The figure below illustrates schematically the code design. It shows how the different classes are organized and how they interact with each other.

<img title="code design" alt="code design" src="./images/code_design.png" />

Each file provides either a set of functions, a namespace or a data container (class).

| File              | Where                 | Description                                                                      |
|-------------------|-----------------------|----------------------------------------------------------------------------------|
| Diagnostics       | `src/common`          | Function to perform diagnostic output                                            |
| ElectroMagn       | `src/common`          | Class that provide an electromagnetic and current grids based on Kokkos 3D views |
| Headers           | `src/common`          | Some helpers                                                                     |
| Managers          | `src/common`          | Free functions called by the subdomain manager, to call the operators            |
| Operators         | `src/common`          | Free functions performing mathematical operations for the simulation             |
| Params            | `src/common`          | Parameters of the simulation                                                     |
| Particle          | `src/common`          | Class that provides a particle container based on Kokkos 1D views                |
| Setup             | `src/common`          | Free function that returns a `Params` object describing a specific setup         |
| SubDomain         | `src/common`          | Data structure that stores and manages a subdomain                               |
| Timers            | `src/common`          | Class that provide timer functionality                                           |
| Tools             | `src/common`          | Various tools for the project                                                    |
| Main              | `src`                 | Main source file for the global code structure                                   |
| Managers          | Implementation folder | Free functions called by the subdomain manager, to call the operators            |
| Operators         | Implementation folder | Free functions performing mathematical operations for the simulation             |
| Name of the setup | `src/setups`          | Setup describing a specific simulation                                           |

## Macros

| Macros                             | Description                                      |
|------------------------------------|--------------------------------------------------|
| `MINI_MINIPIC_DEBUG`               | Enable verbose output                            |
| `MINI_MINIPIC_KOKKOS_SCATTER_VIEW` | Use Kokkos scatter views for projection operator |

## Command-line arguments

The (mini)miniPIC binary accepts the following arguments:

| Short argument | Long argument          | Value type             | Description                                  |
|----------------|------------------------|------------------------|----------------------------------------------|
| `-h`           | `--help`               |                        | print the help page for command line options |
| `-it`          | `--iterations`         | `int`                  | change the number of iterations              |
| `-dmin`        | `--domain_min`         | `double double double` | change the domain minimum boundaries         |
| `-dmax`        | `--domain_max`         | `double double double` | change the domain maximum boundaries         |
| `-rs`          | `--random_seed`        | `int`                  | seed for random generator                    |
| `-pp`          | `--print_period`       | `int`                  | iteration period for terminal printing       |
| `-stp`         | `--save_timers_period` | `int`                  | iteration period for timers saving           |
| `-sts`         | `--save_timers_start`  | `int`                  | iteration start for timers saving            |

Note that command line options overwrite setup parameters.

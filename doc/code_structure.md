# Developer zone

## Domain decomposition

MiniPIC does not support distributed memory parallelism and contains a single domain.

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
| Headers           | `src/common`          | Determine the best headers to use depending on the selected backend              |
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

| Macros                   | Description                                                            |
|--------------------------|------------------------------------------------------------------------|
| `__MINIPIC_KOKKOS_SCATTERVIEW__` | Activate specific KOKKOS projection types: scatter_view.       |
| `__MINIPIC_KOKKOS_UNIFIED__ ` | Macro for code using Kokkos with unified memory                   |

# Developer zone

## Domain decomposition

MiniPIC does not support distributed memory parallelism and contains a single domain.

## PIC loop steps

<img title="pic loop" alt="pic loop" src="./images/pic_loop.png" height="500">

## Code design

The figure below illustrates schematically the code design. It shows how the different classes are organized and how they interact with each other.

<img title="code design" alt="code design" src="./images/code_design.png" height="700">

Each file provides either a set of functions, a namespace or a data container (class).

| File        | Where                  | Description                                                                                               |
|-------------|------------------------|-----------------------------------------------------------------------------------------------------------|
| Headers     | `src/common`           | Determine the best headers to use depending on the selected backend                                       |
| Particle    | `src/common`           | Class that provides a particle container based on Kokkos 1D views                                         |
| ElectroMagn | `src/common`           | Class that provide an electromagnetic and current grids based on Kokkos 3D views                          |
| Diagnostics | `src/common`           | Function to perform diagnostic output                                                                     |
| Timers      | `src/common`           | Class that provide timer functionality                                                                    |
| Main        | `src`                  | Main source file for the global code structure                                                            |
| PICLoop     | implementation folders | Data container representing a domain piece                                                                |
| Operators   | implementation folders | Functions to perform the Particle-In-Cell loop operations (such as interpolator, pusher, projection, etc) |

## Macros

| Macros                   | Description                                                            |
|--------------------------|------------------------------------------------------------------------|
| `__MINIPIC_KOKKOS_SCATTERVIEW__` | Activate specific KOKKOS projection types: scatter_view.       |
| `__MINIPIC_KOKKOS_UNIFIED__ ` | Macro for code using Kokkos with unified memory                   |

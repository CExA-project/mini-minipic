# (mini)miniPIC for CExA

<center>

![(mini)miniPIC title](doc/images/title.png)

</center>

## Presentation

(mini)miniPIC is a playground for computer science and HPC experiments applied to the Particle-In-Cell method.
The project is based on the [miniPIC](https://github.com/Maison-de-la-Simulation/miniPIC) solver.

> [!WARNING]
> (mini)miniPIC is not a code intended to be used for numerical simulation of physical cases.

## Directions

You have in this repo a realistic PIC solver which is at the beginning of being ported to Kokkos.
The goal of the hakathon is to finish this port, and to optimize it for GPU architectures.

Kokkos data containers, program structures, and command line management are already implemented in `src/common` files, you should not have to modify them.
Cases, also known as setups, are hard-coded in `src/setups` and selected at compile time, you should not have to modify them either (at least for porting).
Your work will essentially take place in the `src/exercise` directory, in two files of different granularity.
The operators file is where most operators remain to be ported to Kokkos, whereas the managers file calls the aforementioned operators for one iteration, or for the initialization.

If you want to skip the porting part, you can directly start from the `src/kokkos` code.

<details>

<summary>Some pointers if needed</summary>

[![pointers](https://imgs.xkcd.com/comics/pointers.png) XKCD](https://xkcd.com/138/)

- Start by reading the doc!
- You should first finish to port the operators in `src/exercise`;
- You should port one operator at a time, and running `mini-run` to check if the program is still valid;
- You may have to take care of data location in the process;
- When it's done, you should focus on optimization;
- You should use [Kokkos-tools](https://github.com/kokkos/kokkos-tools) to create regions and breakdown the time spent in the solver;
- [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) may be useful to visualize the regions;
- [Nsight Compute](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html) may be useful to analyze a specific kernel;
- You should identify a bottleneck, try optimization strategies, and measure the gain, rinse and repeat (APOD method: Assess, Parallelize, Optimize, and Deploy).

</details>

## Repository structure

- `doc`: documentation pages;
- `src`: C++ sources:
  - `setups`: headers used to initialize the physical parameters;
  - `common`: source files common to all backends;
  - implementation specific folder (`kokkos`, `exercise`, etc.):
    - `Operators.cpp`: Operator functions;
    - `Managers.cpp`: Functions that call operators for one iteration or for domain initialization;
- `libminipic`: Python libraries for (mini)miniPIC Python tools and validation scripts:
- `script`: Python scripts to read and plot diags (requires `libminipic`);
- `slurm`: Slurm scripts for various supercomputers;
- `external`: Git submodules location.

## Documentation for the exercise

- [Code structure](./doc/code_structure.md)
- [Compilation](./doc/compilation.md)
- [Setups and how to create them](./doc/setups.md)
- [Python tools for validation](./doc/python_tools.md)
- [plot diags](./doc/diags.md)
- [Timers](./doc/timers.md)

## Publications

- SILVA-CUEVAS, J. J., ZYCH, M., PEYEN, K., et al. Towards a complete task-based implementation of a 3D Particle-In-Cell code: performance studies and benchmarks. Computer Physics Communications, 2025, p. 109647. https://doi.org/10.1016/j.cpc.2025.109647

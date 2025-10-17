# miniPIC for CExA

## Presentation

miniPIC is a playground for computer science and HPC experiments applied to the Particle-In-Cell method.

> [!WARNING]
> miniPIC is not a code intended to be used for numerical simulation of physical cases.

### CPU backends

<table>
    <tr>
        <th></th>
        <th style="background-color: #CDD8E0; color: black">Internal Backend</th>
        <th style="background-color: #CDD8E0; color: black">Intel CPU</th>
        <th style="background-color: #CDD8E0; color: black">AMD CPU</th>
        <th style="background-color: #CDD8E0; color: black">ARM based CPU</th>
    </tr>
    <tr>
        <td>OpenMP loop</td>
        <td></td>
        <td style="background-color: #B1E0CB">LLVM, GCC, OneAPI</td>
        <td style="background-color: #B1E0CB">LLVM, GCC, OneAPI</td>
        <td style="background-color: #B1E0CB">LLVM 14, GCC 10</td>
    </tr>
    <tr>
        <td>Kokkos</td>
        <td>OpenMP</td>
        <td style="background-color: #B1E0CB">LLVM, GCC, OneAPI</td>
        <td style="background-color: #E0B4B2">Not tested</td>
        <td style="background-color: #B1E0CB">LLVM, GCC</td>
    </tr>
</table>

### GPU backends

<table>
    <tr>
        <th></th>
        <th style="background-color: #CDD8E0; color: black">Internal Backend</th>
        <th style="background-color: #CDD8E0; color: black">NVIDIA GPU</th>
        <th style="background-color: #CDD8E0; color: black">AMD GPU</th>
        <th style="background-color: #CDD8E0; color: black">Intel GPU</th>
    </tr>
    <tr>
        <td>Kokkos</td>
        <td>CUDA, HIP, SYCL</td>
        <td style="background-color: #B1E0CB">CUDA 12 (tested on V100, A100, H100, GH200)</td>
        <td style="background-color: #B1E0CB">HIP (tested on MI250, MI300)</td>
        <td style="background-color: #B1E0CB">Tested on Intel MAX 1550</td>
    </tr>


</table>

## Repository structure

- `doc`: documentation pages
- `src`: C++ sources
  - `setups`: headers used to initialize the physical parameters
  - `common`: source files common to all backends
  - implementation specific folder (`kokkos`, `exercise`, etc)
    - `Operators.hpp`: Operator functions
    - `SubDomain.hpp`: Operator calls in the time loop
- `libminipic`: Python libraries for miniPIC Python tools and validation scripts
- `script`: Python scripts to read and plot diags
- `slurm`: Slurm scripts for various supercomputers
- `external`: Git submodules location


## How to use miniPIC

- [Compilation](./doc/compilation.md)
- [Understand and create your own setups](./doc/setups.md)
- [plot diags](./doc/diags.md)
- [Timers](./doc/timers.md)
- [Python tools](./doc/python_tools.md)

## How to contribute

- [Continuous Integration](./doc/ci.md)
- [Code structure](./doc/code_structure.md)

## Publications

- SILVA-CUEVAS, J. J., ZYCH, M., PEYEN, K., et al. Towards a complete task-based implementation of a 3D Particle-In-Cell code: performance studies and benchmarks. Computer Physics Communications, 2025, p. 109647. https://doi.org/10.1016/j.cpc.2025.109647

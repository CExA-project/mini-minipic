# Compilation

## Prerequisites

miniPIC uses CMake (version ≥ 3.16) as a build system. For using Kokkos, you have two options:

1. **Using a Git submodule** (recommended)
2. **Using an installed Kokkos library** 

### Method 1: Using a Git submodule (recommended)

```bash
git clone --recurse-submodules <url of the repo>
```

If you have cloned without the submodules:

```sh
git submodule update --init
```

### Method 2: Using an installed Kokkos

If you have Kokkos already installed, you need to tell CMake where to find it.
Either way:

- `-DKokkos_ROOT=/path/to/kokkos/install`: specify Kokkos installation directory
- `-DCMAKE_PREFIX_PATH=/path/to/kokkos/install`: add to CMake search paths

## Build

### Basic compilation (sequential mode)

```bash
cmake -B build
cmake --build build
```

<img title="Warning" alt="Warning" src="./images/warning.png" height="20"> Building in the root directory is not supported.

<img title="Warning" alt="Warning" src="./images/warning.png" height="20"> By default, the code is compiled with Kokkos serial backend (sequential mode).

### Options

CMake useful options:

- `-DCMAKE_CXX_COMPILER=<compiler choice>`: specify the compiler to use;
- `-DCMAKE_BUILD_TYPE=<build type>`: specify the build (most commons are `Debug` and `Release`).

Others:

- `-DMINIPIC_DEBUG=<ON/OFF>`: enable/disable debug messages (`OFF` by default);
- `-DMINIPIC_WARNING=<ON/OFF>`: enable/disable compiler warnings (`OFF` by default);
- `-DMINIPIC_UNIFIED_MEMORY=<ON/OFF>`: enable/disable unified memory views (`OFF` by default);
- `-DMINIPIC_IMPLEMENTATION=<implementation>`: which implementation to use (`exercise` or `kokkos`, default to the former);
- `-DMINIPIC_SETUP=<setup>`: which setup to build and run with (`antenna`, `b_cst`, `beam`, `e_cst`, `thermal`, default to the former).

## Examples

### Using a Git submodule

When using the submodule you can pass directly Kokkos options. See the [Kokkos CMake options documentation](https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html).

#### Basic compilation (defaults to serial backend)

```bash
cmake -B build
cmake --build build
```

#### OpenMP compilation using GCC and OpenMP

```bash
cmake -B build -DCMAKE_CXX_COMPILER=g++ -DKokkos_ENABLE_OPENMP=ON
cmake --build build
```

#### Kokkos compilation using Clang for CPU

```bash
cmake -B build -DCMAKE_CXX_COMPILER=clang++ -DKokkos_ENABLE_SERIAL=ON
cmake --build build
```

#### Kokkos compilation using `nvcc` for Nvidia V100

```bash
cmake -B build -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_VOLTA70=ON
cmake --build build
```

### Using and installed Kokkos

When using a pre-installed Kokkos, you need to specify its location and the configuration should already be set in the compiled and installed library:

- Using Kokkos installed in a custom location

```bash
cmake -B build -DKokkos_ROOT=/path/to/kokkos/install
cmake --build build
```

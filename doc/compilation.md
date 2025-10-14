# Compilation

## Prerequisites

miniPIC uses CMake as a build system. For using Kokkos, you have two options:

1. **Using the git submodule** (recommended)
2. **Using an installed Kokkos library** 

### Method 1: Using the git submodule (recommended)

```bash
git clone --recurse-submodules ...
```

### Method 2: Using an installed Kokkos

If you have Kokkos already installed, you need to tell CMake where to find it:

```bash
git clone ...
```

**Finding Kokkos installation:**
- `-DKokkos_ROOT=/path/to/kokkos/install`: specify Kokkos installation directory
- `-DCMAKE_PREFIX_PATH=/path/to/kokkos/install`: add to CMake search paths

### Basic compilation (sequential mode)

```bash
mkdir build
cd build
cmake ../ 
make
```

<img title="Warning" alt="Warning" src="./images/warning.png" height="20"> Building in the root directory is not supported.

<img title="Warning" alt="Warning" src="./images/warning.png" height="20"> By default, the code is compiled with Kokkos serial backend (sequential mode).

## Options

CMake useful options:

- `-DCMAKE_CXX_COMPILER=<compiler choice>`: specify the compiler to use
- `-DCMAKE_BUILD_TYPE=<build type>`

Others:

- `-DDEBUG=ON/OFF`: enable/disable debug mode (`OFF` by default)
- `-DTEST=ON/OFF`: enable/disable tests mode (for CI, `OFF` by default)
- `-DWARNING=ON/OFF`: enable/disable warnings (`OFF` by default)

## Examples

### Example using installed Kokkos

When using a pre-installed Kokkos, you need to specify its location and the configuration should already be set in the compiled and installed library:

- Using Kokkos installed in a custom location
```bash
cmake ../ -DKokkos_ROOT=/path/to/kokkos/install
make
```

### Example when using git submodule Kokkos

When using the submodule you can pass directly the Kokkos options. See the [Kokkos CMake options documentation](https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html).

After cloning the repository, be sure submodules are initialized:
```bash
 git submodule update --init
```

- Basic compilation (defaults to serial backend)
```bash
cmake ../ 
make
```

- OpenMP compilation using g++

```bash
cmake ../ -DCMAKE_CXX_COMPILER=g++ -DKokkos_ENABLE_OPENMP=ON
make
```

- Kokkos compilation using clang++ for CPU
```bash
cmake ../ -DCMAKE_CXX_COMPILER=clang++ -DKokkos_ENABLE_SERIAL=ON 
make
```

- Kokkos compilation using nvcc for Nvidia V100

```bash
cmake ../ -DCMAKE_CXX_COMPILER=nvcc -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_VOLTA70=ON
make
```
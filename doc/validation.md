# Validation

## Benchmark validation

A validation mechanism can be used locally and on super-computer to validate the code modification.
A python script `run.py` located in the `tests` folder is designed to test the whole code against reference data for a list of benchmarks.
This script needs Python libraries located in `lib`.

You can access the help page by doing:

```bash
python tests/run.py -h
```

### Available options:

Here is a list of available options:

| Option | Long Option | Description |
| --- | --- | --- |
| `-h` | `--help` | Show this help message and exit |
| `-g CONFIG` | `--config CONFIG` | Configuration choice: cpu (default), gpu |
| `-c COMPILER` | `--compiler COMPILER` | Custom compiler choice |
| `-b BENCHMARKS` | `--benchmarks BENCHMARKS` | Specific benchmark, you can specify several benchmarks with a coma. For instance "default,beam" |
| | `--build-dir` | Build directory to use, default to `build` |
| `-a ARGUMENTS` | `--arguments ARGUMENTS` | Default arguments |
| | `--fresh` | Whether to delete or already existing files |
| | `--clean` | Whether to delete or not the generated files |
| | `--no-evaluate` | If used, do not evaluate against the reference |
| | `--compile-only` | If used, only compile the tests |
| | `--threshold THRESHOLD` | Threshold for the validation |
| | `--save-timers` | Save the timers for each benchmark |
| | `--env` | Custom environment variables for the execution |
| | `--cmake-args` | Custom CMake arguments |
| | `--cmake-args-add` | Append custom CMake arguments |

### Configurations

Here is a list of all configurations:

| Configuration | Description |
| --- | --- |
| cpu-serial | CPU serial |
| cpu-openmp | CPU with OpenMP, 8 threads |
| gpu-v100 | GPU on V100 |
| gpu-a100 | GPU on A100 |
| gpu-h100 | GPU on H100 |
| gpu-mi250 | GPU on MI250 |
| gpu-mi300a | GPU on MI300A, with `HSA_XNACK` set |

### Usage examples

#### Default run

```bash
python tests/run.py
```

#### Specific configuration

```bash
python tests/run.py -g gpu
```

#### Custom compiler

```bash
python tests/run.py -c clang++
```

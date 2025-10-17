# Python tools (libminipic)

MiniPIC uses several tools gathered in the `libminipic` library.

Install this library and its dependencies locally with:

```sh
pip install --user .
```

Even better, install it in a virtual environment.

## `mini-run`

The command `mini-run` is used to build miniPIC for a selection of setups, to execute it and to validate its results.
For validation, `mini-run` calls the same functions as `mini-validate` internally.

You can get some help with:

```bash
mini-run -h
```

### Available options:

Here is a list of the available options:

| Option | Long Option | Description |
| --- | --- | --- |
| `-h` | `--help` | Show a help message and exit |
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

## `mini-validate`

A validation mechanism can be used to validate the code outputs after it ran.
Output files are checked against reference values.

You can access the help page by doing:

```bash
mini-validate -h
```

### Available options:

Here is a list of the available options:

| Option | Long Option | Description |
| --- | --- | --- |
| `setup` | | Name of the setup |
| | `--threshold THRESHOLD` | Threshold for the validation |

### Setups

The possible setups are:

- `antenna`
- `bcst`
- `ecst`
- `beam`
- `thermal`

For a given benchmark, the output results can be analyzed to validate the run.
This only happens if a script of the same name is present in the directory `validate` on the root of the repository.
For instance, `default.py` being implemented in `validate`, the run of `default.hpp` will be analyzed at the end of the simulation.

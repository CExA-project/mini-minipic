"""Runs tests for CI."""

# ________________________________________________________________________________
# Imports

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time

from libminipic.ci import print_command, print_step
from libminipic.validate import THRESHOLD, validate_setup

# ________________________________________________________________________________
# Parameters

# Configurations for running the tests
configuration_list = {
    "cpu-serial": {
        "compiler": "g++",
        "cmake": [
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        "env": {},
        "prefix": [],
        "exe_name": "mini-minipic",
        "setups": ["thermal", "beam", "antenna", "e_cst", "b_cst"],
        "args": [[], [], []],
    },
    "cpu-openmp": {
        "compiler": "g++",
        "cmake": [
            "-DCMAKE_BUILD_TYPE=Release",
            "-DKokkos_ENABLE_OPENMP=ON",
        ],
        "env": {
            "OMP_PROC_BIND": "spread",
            "OMP_PLACES": "threads",
            "OMP_NUM_THREADS": "8",
        },
        "prefix": [],
        "exe_name": "mini-minipic",
        "setups": ["thermal", "beam", "antenna", "e_cst", "b_cst"],
        "args": [[], [], []],
    },
    "gpu-v100": {
        "compiler": "g++",
        "cmake": [
            "-DCMAKE_BUILD_TYPE=Release",
            "-DKokkos_ENABLE_CUDA=ON",
            "-DKokkos_ARCH_VOLTA70=ON",
        ],
        "env": {},
        "prefix": [],
        "exe_name": "mini-minipic",
        "setups": ["thermal", "beam", "antenna"],
        "args": [[], [], []],
    },
    "gpu-a100": {
        "compiler": "g++",
        "cmake": [
            "-DCMAKE_BUILD_TYPE=Release",
            "-DKokkos_ENABLE_CUDA=ON",
            "-DKokkos_ARCH_AMPERE80=ON",
        ],
        "env": {},
        "prefix": [],
        "exe_name": "mini-minipic",
        "setups": ["thermal", "beam", "antenna"],
        "args": [[], [], []],
    },
    "gpu-h100": {
        "compiler": "g++",
        "cmake": [
            "-DCMAKE_BUILD_TYPE=Release",
            "-DKokkos_ENABLE_CUDA=ON",
            "-DKokkos_ARCH_HOPPER90=ON",
        ],
        "env": {},
        "prefix": [],
        "exe_name": "mini-minipic",
        "setups": ["thermal", "beam", "antenna"],
        "args": [[], [], []],
    },
    "gpu-mi250": {
        "compiler": "hipcc",
        "cmake": [
            "-DCMAKE_BUILD_TYPE=Release",
            "-DKokkos_ENABLE_HIP=ON",
            "-DKokkos_ARCH_AMD_GFX90A=ON",
        ],
        "env": {},
        "prefix": [],
        "exe_name": "mini-minipic",
        "setups": ["thermal", "beam", "antenna"],
        "args": [[], [], []],
    },
    "gpu-mi300a": {
        "compiler": "hipcc",
        "cmake": [
            "-DCMAKE_BUILD_TYPE=Release",
            "-DKokkos_ENABLE_HIP=ON",
            "-DKokkos_ARCH_AMD_GFX942=ON",
        ],
        "env": {"HSA_XNACK": "1"},
        "prefix": [],
        "exe_name": "mini-minipic",
        "setups": ["thermal", "beam", "antenna"],
        "args": [[], [], []],
    },
}


def run():
    config_description = "List of all configurations: \n\n"

    for config, config_dict in configuration_list.items():
        config_description += "> {}: \n".format(config)
        config_description += "    - compiler: {}\n".format(config_dict["compiler"])
        config_description += "    - cmake options: {}\n".format(config_dict["cmake"])
        config_description += "    - run prefixes: {}\n".format(config_dict["prefix"])
        config_description += "    - run env: {}\n".format(config_dict["env"])

    config_description += "    \n\n"

    # Command line arguments
    parser = argparse.ArgumentParser(
        description="Custom arguments for the validation process",
        epilog=config_description,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-g",
        "--config",
        default="cpu-openmp",
        help="configuration choice: {}, default to cpu-openmp".format(
            ", ".join(configuration_list.keys())
        ),
    )
    parser.add_argument("-c", "--compiler", help="custom compiler choice")
    parser.add_argument(
        "-s",
        "--setups",
        help=(
            "specific setup, you can specify several setups with a coma, "
            'for instance "default,beam"'
        ),
        default=None,
    )
    parser.add_argument(
        "--build-dir", help="build directory, default to 'build'", default="build"
    )
    parser.add_argument(
        "--implementation",
        help=(
            "which implementation to use, 'kokkos' or 'exercise', default to 'exercise'"
        ),
        default="exercise",
    )
    parser.add_argument(
        "-j",
        "--parallel",
        help="number of jobs to use for the compilation, default to 4",
        type=int,
        default=4,
    )
    parser.add_argument("-a", "--arguments", help="default arguments", default=None)
    parser.add_argument(
        "--fresh",
        help="whether to delete or not already existing files",
        action="store_true",
    )
    parser.add_argument(
        "--clean",
        help="whether to delete or not the generated files",
        action="store_true",
    )
    parser.add_argument(
        "--no-evaluate",
        help="if used, do not evaluate against the reference",
        action="store_true",
    )
    parser.add_argument(
        "--compile-only", help="if used, only compile the tests", action="store_true"
    )
    parser.add_argument(
        "--threshold",
        help="threshold for the validation",
        default=THRESHOLD,
        type=float,
    )
    parser.add_argument(
        "--save-timers", help="save the timers for each setup", action="store_true"
    )
    parser.add_argument(
        "--prefix",
        help="add custom prefix for the execution, for instance srun",
    )
    parser.add_argument(
        "-p",
        "--path",
        help="path where the CMakeLists.txt is, default to corrent working directory",
        default=os.getcwd(),
    )
    parser.add_argument(
        "--env",
        help="add custom environment variables for the execution, for instance `OMP_PROC_BIND=spread`",
    )
    parser.add_argument(
        "--cmake-args", help="set custom cmake arguments for the compilation"
    )
    parser.add_argument(
        "--cmake-args-add", help="add custom cmake arguments for the compilation"
    )

    args = parser.parse_args()

    # Selected configuration
    configuration = args.config

    selected_config = configuration_list[configuration].copy()

    # Select compiler
    if args.compiler:
        selected_config["compiler"] = args.compiler

    clean = args.clean

    # Evaluate or not
    evaluate = not args.no_evaluate

    # Compile only
    compile_only = args.compile_only

    # Save timers
    save_timers = args.save_timers

    # Select setup list
    if args.setups:

        selected_config["setups"] = []

        for b in args.setups.split(","):
            selected_config["setups"].append(b)

    # Environment
    if args.env:
        for local_env in args.env.split():
            key, value = local_env.split("=")
            selected_config["env"][key] = value

    # Prefix
    if args.prefix:
        selected_config["prefix"].extend(args.prefix.split())

    # Select arguments
    if args.arguments:

        if "," in args.arguments:

            selected_config["args"] = []
            for arg in args.arguments.split(","):
                selected_config["args"].append(arg)

        else:

            selected_config["args"] = []
            for _ in range(len(selected_config["setups"])):
                selected_config["args"].append(args.arguments)

    else:

        selected_config["args"] = []
        for _ in range(len(selected_config["setups"])):
            selected_config["args"].append("")

    # Set custom cmake arguments
    if args.cmake_args:

        selected_config["cmake"] = shlex.split(args.cmake_args)

    # Add custom cmake arguments
    if args.cmake_args_add:
        selected_config["cmake"].extend(shlex.split(args.cmake_args_add))

    # threshold
    threshold = args.threshold

    fresh = args.fresh

    assert threshold > 0, "Threshold should be positive"

    # Get local path
    root_dir = args.path
    build_dir = os.path.join(root_dir, args.build_dir)

    # Add validation path to the PYTHONPATH
    sys.path.append("{}/validation".format(root_dir))

    # Add root path to the PYTHONPATH
    sys.path.append("{}".format(root_dir))

    # Terminal size

    # Get the git hash in variable git_hash
    try:
        git_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], timeout=60)
            .strip()
            .decode("utf-8")
        )
    except:
        git_hash = "unknown"

    # Get the git branch in variable git_branch
    try:
        git_branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], timeout=60
            )
            .strip()
            .decode("utf-8")
        )

        # get branch from detatched head
        if git_branch == "HEAD":
            git_branch = "detached"
    except:
        git_branch = "unknown"

    # number of jobs for parallel build
    jobs = args.parallel

    # Get the pipeline id in variable pipeline_id
    pipeline_id = os.environ.get("CI_PIPELINE_ID", "unknown")

    # Create unique id of the form yyyymmdd_hh_mm_ss_pipeline_id_git_hash
    unique_id = time.strftime("%Y%m%d_%H:%M:%S") + "_" + pipeline_id + "_" + git_hash

    # Print some info
    print_step("Summary", level=0)

    print(" Git branch: {}".format(git_branch))
    print(" Git hash: {}".format(git_hash))
    print(" Pipeline id: {}".format(pipeline_id))
    print(" Root dir: {}".format(root_dir))
    print(" Build dir: {}".format(build_dir))
    print(" Selected configuration: {}".format(configuration))
    print(" Compiler: {}".format(selected_config["compiler"]))
    print(" Fresh: {}".format(fresh))
    print(" Clean: {}".format(clean))
    print(" Save: {}".format(save_timers))
    print(" Unique id: {}".format(unique_id))
    if compile_only:
        print(" Compile only: {}".format(compile_only))
    else:
        print(" Evaluate: {}".format(evaluate))
    print(" Threshold: {}".format(threshold))
    print(" Prefix: {}".format(selected_config["prefix"]))
    print(" Env: {}".format(selected_config["env"]))
    print(" Cmake args: {}".format(selected_config["cmake"]))

    # print all benchamrks

    print(" Setups: ")
    for setup_id, setup in enumerate(selected_config["setups"]):
        print("   - {} with args '{}'".format(setup, selected_config["args"][setup_id]))

    # Run setups

    # Get configuration
    compiler = selected_config["compiler"]
    cmake = selected_config["cmake"]
    executable_name = selected_config["exe_name"]
    implementation = args.implementation

    for setup_id, setup in enumerate(selected_config["setups"]):

        env = selected_config["env"]
        prefix = selected_config["prefix"]
        args = selected_config["args"][setup_id]

        # bench parameters

        bench_dir = os.path.join(build_dir, setup)

        print_step(f"Setup {setup}", level=0)

        # ____________________________________________________________________________
        # Configuration

        # Create a directory for this setup
        if fresh:
            shutil.rmtree(bench_dir, ignore_errors=True)
        os.makedirs(bench_dir, exist_ok=True)

        cmake_command = ["cmake", root_dir, "-DMINI_MINIPIC_SETUP={}".format(setup)]

        # handle implementation
        if implementation:
            cmake_command.append(f"-DMINI_MINIPIC_IMPLEMENTATION={implementation}")

        # handle compiler
        if compiler:
            cmake_command.append("-DCMAKE_CXX_COMPILER={}".format(compiler))

        cmake_command.extend(cmake)

        print_step("Configuration")
        print_command(cmake_command)

        subprocess.run(cmake_command, cwd=bench_dir, check=True)

        # ____________________________________________________________________________
        # Compilation

        make_command = ["cmake", "--build", bench_dir, "--parallel", str(jobs)]

        print_step("Compilation")
        print_command(make_command)

        subprocess.run(make_command, check=True)

        # Check
        exe_exists = os.path.exists(os.path.join(bench_dir, executable_name))
        assert exe_exists, "Executable not generated"

        # ____________________________________________________________________________
        # Execution

        if not compile_only:

            current_env = env.copy()

            print_step("Execution")

            run_command = [
                "./{}".format(executable_name),
                *args,
            ]  # srun numactl --interleave=all
            if prefix:
                run_command = [*prefix, *run_command]

            print_command(run_command, env=current_env)

            subprocess.run(
                run_command,
                check=False,
                cwd=bench_dir,
                env={**(os.environ), **current_env},
            )

            # ____________________________________________________________________________
            # Check results

            print_step("Validation")
            print_command(
                [
                    "mini-validate",
                    "--path",
                    bench_dir,
                    "--setup",
                    setup,
                    "--threshold",
                    threshold,
                ]
            )

            validate_setup(bench_dir, setup, threshold)

            # ____________________________________________________________________________
            # Read timers (json format)

            if save_timers:

                print_step("Timers")

                with open(os.path.join(bench_dir, "timers.json")) as f:
                    raw_timers_dict = json.load(f)

                pic_it_time = raw_timers_dict["final"]["main loop (no diags)"][0]

                ci_file_name = os.path.join(
                    build_dir, "ci_{}_{}_timers.json".format(setup, configuration)
                )

                if not os.path.exists(ci_file_name):
                    with open(ci_file_name, "w") as f:
                        f.write("{}")

                with open(ci_file_name, "r") as f:
                    ci_timers_dict = json.load(f)

                # Read all ci_timers_dict keys and compute an average time:

                average_pic_it_time = 0.0
                number_of_entry = len(ci_timers_dict.keys())
                for key in ci_timers_dict.keys():
                    average_pic_it_time += ci_timers_dict[key]["times"]["iteration"]

                if number_of_entry > 0:
                    average_pic_it_time /= number_of_entry

                print("    Average iteration time: {}".format(average_pic_it_time))
                print("    Current iteration time: {}".format(pic_it_time))

                # Create the new entry

                ci_timers_dict[unique_id] = {
                    "date": time.strftime("%Y_%m_%d_%H_%M_%S"),
                    "branch": git_branch,
                    "hash": git_hash,
                    "pipeline_id": pipeline_id,
                    "times": {"iteration": pic_it_time},
                }

                # Save the new entry

                with open(ci_file_name, "w") as f:
                    json.dump(ci_timers_dict, f, indent=4)

                print("    Timers saved in {}".format(ci_file_name))

        # ____________________________________________________________________________

        if clean and os.path.exists(bench_dir):

            print_step("Cleanup")

            shutil.rmtree(bench_dir, ignore_errors=True)

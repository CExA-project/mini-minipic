"""Gateway for validation."""

import importlib
import os
import sys
from argparse import ArgumentParser


def validate_setup(path, setup, threshold):
    module = importlib.import_module(f"libminipic.validation.{setup}", None)

    if path:
        os.chdir(path)

    if not os.path.isdir("diags"):
        print("Directory diags should be present where you run this script")
        sys.exit(1)

    module.validate(threshold)

    print(f"\033[32mBenchmark `{setup}` tested with success \033[39m")


def validate():
    parser = ArgumentParser(description="Run a validation script")

    parser.add_argument("setup", help="name of the setup to validate")
    parser.add_argument("--path", help="path to the execution directory")
    parser.add_argument(
        "--threshold", help="threshold for the validation", default=1e-10, type=float
    )

    args = parser.parse_args()

    validate_setup(args.path, args.setup, args.threshold)

"""Gateway for validation."""

import importlib
import os
import re
import sys
from argparse import ArgumentParser
from pathlib import Path

from libminipic.ci import print_success
from libminipic.exceptions import IncorrectFileMiniPICError, MissingFileMiniPICError

CMAKE_CACHE_FILENAME = "CMakeCache.txt"

THRESHOLD = 1e-10


def detect_setup(path: Path) -> str:
    cmake_cache_file = path / CMAKE_CACHE_FILENAME
    if not cmake_cache_file.exists():
        raise MissingFileMiniPICError(f"Cannot find {cmake_cache_file}")

    cmake_cache_content = cmake_cache_file.read_text()
    matcher = re.findall(r"MINIPIC_SETUP:STRING=(.*)", cmake_cache_content)

    if not matcher:
        raise IncorrectFileMiniPICError(f"Cannot find setup in {cmake_cache_file}")

    # return the first element, as we know there is at least one
    return matcher[0].strip()


def validate_setup(path, setup=None, threshold=THRESHOLD):
    if not setup:
        setup = detect_setup(path)
        print(f"Autodetected setup: {setup}")

    module = importlib.import_module(f"libminipic.validation.{setup}", None)

    os.chdir(path)

    if not os.path.isdir("diags"):
        raise MissingFileMiniPICError(f"Directory diags is not in {path}")

    module.validate(threshold)

    print_success(f"Setup {setup} tested with success")


def validate():
    parser = ArgumentParser(description="Run a validation script")

    parser.add_argument(
        "-s", "--setup", help="name of the setup to validate (default to autodetect)"
    )
    parser.add_argument(
        "-p",
        "--path",
        help="path to the execution directory (default to current working directory)",
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument(
        "--threshold",
        help="threshold for the validation",
        default=THRESHOLD,
        type=float,
    )

    args = parser.parse_args()

    validate_setup(args.path, args.setup, args.threshold)

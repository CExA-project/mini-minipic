"""Tools for CI."""

import math
import shutil
import sys

from termcolor import cprint

from libminipic.exceptions import (
    IncorrectValueMiniPICError,
    ThresholdValueMiniPICError,
    ValueMiniPICError,
)

TERMINAL_WIDTH = shutil.get_terminal_size().columns

SEPARATOR_LEVELS = ["=", "-", "."]


def print_success(message):
    """Print a success message."""
    cprint(message, "green", attrs=["bold"])
    force_print()


def print_failure(message):
    """Print a failure message."""
    cprint(message, "red", attrs=["bold"])
    force_print()


def print_step(message, level=1):
    """Print a step message with a line."""
    sep = SEPARATOR_LEVELS[level]
    cprint(
        f"{sep}{sep} {message} {sep}{sep}".ljust(TERMINAL_WIDTH, sep),
        "cyan",
        attrs=["bold"],
    )
    force_print()


def print_command(command, env=None):
    """Print a `subprocess.run` command list."""
    env_list = []
    if env is not None:
        env_list = [f"{k}={v}" for k, v in env.items()]

    cprint(" ".join(str(v) for v in (*env_list, *command)), "yellow")
    force_print()


def force_print():
    """Force flush to see text on time.

    Especially useful on HPC."""
    sys.stdout.flush()


def evaluate(value, reference, threshold, operator="relative", txt=""):

    flag = False
    error_value = 0

    if operator == "<":

        error_value = math.fabs(value - threshold)

        flag = value > threshold

        if flag:
            raise ThresholdValueMiniPICError(
                f"{txt}: {value} > {threshold} with error {error_value}"
            )

        return

    if operator == "==":

        error_value = math.fabs(value - threshold)

        flag = error_value != 0

        if flag:
            raise IncorrectValueMiniPICError(
                f"{txt}: {value} not equal to {threshold} with value {error_value}"
            )

        return

    if operator == "relative":

        if reference == 0:
            raise ValueMiniPICError(
                "Can not evaluate a relative error with reference == 0"
            )

        error_value = math.fabs((value - reference) / reference)

        flag = error_value > threshold

        if flag:
            raise ThresholdValueMiniPICError(
                f"{txt}: {value} vs {reference}, with error {error_value} for relative threshold {threshold}"
            )

        return

    if operator == "absolute":

        error_value = math.fabs(value - reference)

        flag = error_value > threshold

        if flag:
            raise ThresholdValueMiniPICError(
                f"{txt}: {value} vs {reference} with error {error_value} for absolute threshold {threshold}"
            )

        return

    raise ValueMiniPICError(f"Operator not recognized {operator}")

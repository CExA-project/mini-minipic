#!/usr/bin/env python3

import re
import numpy
from subprocess import run
from pathlib import Path
from argparse import ArgumentParser

PIC_ITERATIONS_MATCHER = re.compile(r"PIC iterations \| +(\d+\.\d+) +\|")


def execute(directory: Path) -> float:
    outcome = run(["./mini-minipic"], cwd=directory, capture_output=True, check=True)
    matcher = PIC_ITERATIONS_MATCHER.search(outcome.stdout.decode())
    return float(matcher.group(1))


def execute_all(directory: Path, nb_repeat: int) -> list[float]:
    timings = []
    for repeat_id in range(nb_repeat):
        timing = execute(directory)
        print(f"Run {directory.name} {repeat_id}: {timing} s")
        timings.append(timing)

    return timings


def analyze(timings: list[float]) -> tuple[float, float]:
    mean = numpy.mean(timings)
    std = numpy.std(timings)

    return float(mean), float(std)


if __name__ == "__main__":
    parser = ArgumentParser(description="Run mini-minipic a certain number of times and print stats")
    parser.add_argument("-n", "--nb-repeat", default=5, type=int, help="number of executions")
    parser.add_argument("directory", type=Path, help="directory where mini-minipic is")

    args = parser.parse_args()

    timings = execute_all(args.directory, args.nb_repeat)
    mean, std = analyze(timings)

    print(f"Mean {args.directory.name}: {mean} s")
    print(f"Std {args.directory.name}: {std} s")

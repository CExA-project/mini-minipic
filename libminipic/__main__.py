"""Lib execution from command line."""

import sys

from libminipic.ci import print_failure
from libminipic.exceptions import MiniPICError
from libminipic.run import run
from libminipic.validate import validate


def execute(function):
    """Execute a function and catch usual errors."""

    try:
        function()
        return 0

    except KeyboardInterrupt:
        print_failure("Interrumpted by user")
        return 128

    except MiniPICError as exception:
        print_failure(exception)
        return 2

    except:
        print_failure("Unexpected error")
        raise


def execute_validate():
    sys.exit(execute(validate))


def execute_run():
    sys.exit(execute(run))

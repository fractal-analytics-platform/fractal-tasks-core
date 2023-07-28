# Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Tommaso Comparin <tommaso.comparin@exact-lab.it>
#
# This file is part of Fractal and was originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
"""
Standard input/output interface for tasks.
"""
import json
import logging
from argparse import ArgumentParser
from json import JSONEncoder
from pathlib import Path
from typing import Callable
from typing import Optional


class TaskParameterEncoder(JSONEncoder):
    """
    Custom JSONEncoder that transforms Path objects to strings.
    """

    def default(self, value):
        """
        Subclass implementation of `default`, to serialize Path objects as
        strings.
        """
        if isinstance(value, Path):
            return value.as_posix()
        return JSONEncoder.default(self, value)


def run_fractal_task(
    *,
    task_function: Callable,
    logger_name: Optional[str] = None,
):
    """
    Implement standard task interface and call task_function.

    Args:
        task_function: the callable function that runs the task.
        logger_name: TBD
    """

    # Parse `-j` and `--metadata-out` arguments
    parser = ArgumentParser()
    parser.add_argument(
        "-j", "--json", help="Read parameters from json file", required=True
    )
    parser.add_argument(
        "--metadata-out",
        help="Output file to redirect serialised returned data",
        required=True,
    )
    args = parser.parse_args()

    # Set logger
    logger = logging.getLogger(logger_name)

    # Preliminary check
    if Path(args.metadata_out).exists():
        logger.error(
            f"Output file {args.metadata_out} already exists. Terminating"
        )
        exit(1)

    # Read parameters dictionary
    with open(args.json, "r") as f:
        pars = json.load(f)

    # Run task
    logger.info(f"START {task_function.__name__} task")
    metadata_update = task_function(**pars)
    logger.info(f"END {task_function.__name__} task")

    # Write output metadata to file, with custom JSON encoder
    with open(args.metadata_out, "w") as fout:
        json.dump(metadata_update, fout, cls=TaskParameterEncoder, indent=2)

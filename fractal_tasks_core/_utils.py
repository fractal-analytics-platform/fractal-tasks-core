# Starting from Python 3.9 (see PEP 585) we can use type hints like
# `type[BaseModel`. For versions 3.7 and 3.8, this is available through an
# additional import
from __future__ import annotations

import json
import logging
from argparse import ArgumentParser
from json import JSONEncoder
from pathlib import Path
from typing import Callable
from typing import get_type_hints

from pydantic import BaseModel


class TaskParameterEncoder(JSONEncoder):
    """
    Custom JSONEncoder that transforms Path objects to strings
    """

    def default(self, value):
        if isinstance(value, Path):
            return value.as_posix()
        return JSONEncoder.default(self, value)


def run_fractal_task(
    *,
    task_function: Callable,
    TaskArgsModel: type[BaseModel] = None,
    logger_name: str = None,
):
    """
    Implement standard task interface and call task_function. If TaskArgsModel
    is not None, validate arguments against given model.

    :param task_function: the callable function that runs the task
    :param TaskArgsModel: a class specifying all types for task arguments
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

    if TaskArgsModel is None:
        # Run task without validating arguments' types
        logger.info(f"START {task_function.__name__} task")
        metadata_update = task_function(**pars)
        logger.info(f"END {task_function.__name__} task")
    else:
        # Check match of type hints in task_function and TaskArgsModel
        task_function_type_hints = get_type_hints(task_function)
        task_function_type_hints.pop("return", None)
        TaskArgsModel_type_hints = get_type_hints(TaskArgsModel)
        if task_function_type_hints != TaskArgsModel_type_hints:
            raise ValueError(
                "task_function_type_hints differs from "
                "TaskArgsModel_type_hints. Please check that "
                "they are up-to-date.\n"
                f"{sorted(task_function_type_hints.items())}\n"
                f"{sorted(TaskArgsModel_type_hints.items())}"
            )
        # Validating arguments' types and run task
        task_args = TaskArgsModel(**pars)
        logger.info(f"START {task_function.__name__} task")
        metadata_update = task_function(**task_args.dict())
        logger.info(f"END {task_function.__name__} task")

    # Write output metadata to file, with custom JSON encoder
    with open(args.metadata_out, "w") as fout:
        json.dump(metadata_update, fout, cls=TaskParameterEncoder)

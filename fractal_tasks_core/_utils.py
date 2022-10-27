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
    *, task_function: Callable, TaskArgsModel: type[BaseModel] = None
):
    """
    Implement standard task interface and call task_function. If TaskArgsModel
    is not None, validate arguments against given model.
    """

    # Parse `-j` and `--metadata-out` arguments
    parser = ArgumentParser()
    parser.add_argument("-j", "--json", help="Read parameters from json file")
    parser.add_argument(
        "--metadata-out",
        help="Output file to redirect serialised returned data",
    )
    args = parser.parse_args()

    # Preliminary check
    if Path(args.metadata_out).exists():
        logging.error(
            f"Output file {args.metadata_out} already exists. Terminating"
        )
        exit(1)

    # Read parameters dictionary
    with open(args.json, "r") as f:
        pars = json.load(f)

    if TaskArgsModel is None:
        # Run task without validating arguments' types
        metadata_update = task_function(**pars)
    else:
        # Validating arguments' types and run task
        task_args = TaskArgsModel(**pars)
        metadata_update = task_function(**task_args.dict())

    # Write output metadata to file, with custom JSON encoder
    with open(args.metadata_out, "w") as fout:
        json.dump(metadata_update, fout, cls=TaskParameterEncoder)

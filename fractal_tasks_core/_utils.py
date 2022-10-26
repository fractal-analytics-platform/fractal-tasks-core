import json
import logging
from argparse import ArgumentParser
from json import JSONEncoder
from pathlib import Path


class TaskParameterEncoder(JSONEncoder):
    def default(self, value):
        if isinstance(value, Path):
            return value.as_posix()
        return JSONEncoder.default(self, value)


def run_fractal_task(callable_function, args_model=None):
    parser = ArgumentParser()
    parser.add_argument("-j", "--json", help="Read parameters from json file")
    parser.add_argument(
        "--metadata-out",
        help=(
            "Output file to redirect serialised returned data "
            "(default stdout)"
        ),
    )

    args = parser.parse_args()

    if args.metadata_out and Path(args.metadata_out).exists():
        logging.error(
            f"Output file {args.metadata_out} already exists. Terminating"
        )
        exit(1)

    pars = {}
    if args.json:
        with open(args.json, "r") as f:
            pars = json.load(f)

    if args_model is not None:
        task_args = args_model(**pars)
        metadata_update = callable_function(**task_args.dict())
    else:
        metadata_update = callable_function(**pars)

    if args.metadata_out:
        with open(args.metadata_out, "w") as fout:
            json.dump(metadata_update, fout, cls=TaskParameterEncoder)

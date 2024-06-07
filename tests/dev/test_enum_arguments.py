import json
from enum import Enum

from devtools import debug
from pydantic.decorator import validate_arguments

from fractal_tasks_core.dev.lib_args_schemas import (
    create_schema_for_single_task,
)


class ColorA(Enum):
    RED = "this-is-red"
    GREEN = "this-is-green"


ColorB = Enum(
    "ColorB",
    {"RED": "this-is-red", "GREEN": "this-is-green"},
    type=str,
)


@validate_arguments
def task_function(
    arg_A: ColorA,
    arg_B: ColorB,
):
    """
    Short task description

    Args:
        arg_A: Description of arg_A.
        arg_B: Description of arg_B.
    """
    pass


def test_enum_argument():
    """
    This test only asserts that `create_schema_for_single_task` runs
    successfully. Its goal is also to offer a quick way to experiment
    with new task arguments and play with the generated JSON Schema,
    without re-building the whole fractal-tasks-core manifest.
    """
    schema = create_schema_for_single_task(
        task_function=task_function,
        executable=__file__,
        package=None,
        verbose=True,
    )
    debug(schema)
    print(json.dumps(schema, indent=2))

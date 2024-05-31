import json
from enum import Enum

from devtools import debug
from pydantic.decorator import validate_arguments

from fractal_tasks_core.dev.lib_args_schemas import (
    create_schema_for_single_task,
)
from fractal_tasks_core.tasks.io_models import InitArgsRegistration


class Color(Enum):
    RED = 1
    GREEN = 2


# @validate_arguments
def task_function_1(
    arg1: InitArgsRegistration,
):
    """
    Short task description

    Args:
        arg1: Description of arg1.
    """
    pass


@validate_arguments
def task_function_2(
    arg2: Color,
):
    """
    Short task description

    Args:
        arg2: Description of arg2.
    """
    pass


def test_pydantic_argument():
    schema = create_schema_for_single_task(
        task_function=task_function_1,
        executable=__file__,
        package=None,
        verbose=True,
    )
    debug(schema)
    print(json.dumps(schema, indent=2))


def test_enum_argument():
    schema = create_schema_for_single_task(
        task_function=task_function_2,
        executable=__file__,
        package=None,
        verbose=True,
    )
    debug(schema)
    print(json.dumps(schema, indent=2))

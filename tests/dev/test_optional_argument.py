import json
from typing import Optional

from devtools import debug
from pydantic.v1.decorator import validate_arguments
from pydantic.validate_call_decorator import validate_call

from fractal_tasks_core.dev.lib_args_schemas import (
    create_schema_for_single_task,
)


@validate_arguments
def task_function_1(
    argument: Optional[str] = None,
):
    """
    Short task description

    Args:
        argument: This is the argument description
    """
    pass


@validate_call
def task_function_2(
    argument: Optional[str] = None,
):
    """
    Short task description

    Args:
        argument: This is the argument description
    """
    pass


def test_optional_argument_1():
    schema = create_schema_for_single_task(
        task_function=task_function_1,
        executable=__file__,
        package=None,
        args_schema_version="pydantic_v1",
        verbose=True,
    )
    debug(schema)
    print(json.dumps(schema, indent=2))


def test_optional_argument_2():
    schema = create_schema_for_single_task(
        task_function=task_function_2,
        executable=__file__,
        package=None,
        args_schema_version="pydantic_v2",
        verbose=True,
    )
    debug(schema)
    print(json.dumps(schema, indent=2))

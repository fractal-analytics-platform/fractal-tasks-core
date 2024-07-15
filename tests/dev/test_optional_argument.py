import json
from typing import Optional

from pydantic.v1.decorator import validate_arguments
from pydantic.validate_call_decorator import validate_call

from fractal_tasks_core.dev.lib_args_schemas import (
    create_schema_for_single_task,
)


def task_function(
    arg1: Optional[str] = None,
    arg2: Optional[list[str]] = None,
):
    """
    Short task description

    Args:
        arg1: This is the argument description
        arg2: This is the argument description
    """
    pass


def test_optional_argument():

    schema1 = create_schema_for_single_task(
        task_function=validate_arguments(task_function),
        executable=__file__,
        package=None,
        args_schema_version="pydantic_v1",
        verbose=True,
    )

    schema2 = create_schema_for_single_task(
        task_function=validate_call(task_function),
        executable=__file__,
        package=None,
        args_schema_version="pydantic_v2",
        verbose=True,
    )

    print(json.dumps(schema1, indent=2, sort_keys=True))
    print(json.dumps(schema2, indent=2, sort_keys=True))
    print()
    assert schema1 == schema2

import json
from typing import Optional

from pydantic.validate_call_decorator import validate_call

from fractal_tasks_core.dev.lib_args_schemas import (
    create_schema_for_single_task,
)


@validate_call
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

    schema = create_schema_for_single_task(
        task_function=validate_call(task_function),
        executable=__file__,
        package=None,
        verbose=True,
    )
    print(json.dumps(schema, indent=2, sort_keys=True))
    print()

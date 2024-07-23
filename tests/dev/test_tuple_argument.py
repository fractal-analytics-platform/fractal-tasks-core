import json

from devtools import debug
from pydantic import validate_call

from fractal_tasks_core.dev.lib_args_schemas import (
    create_schema_for_single_task,
)


@validate_call
def task_function(arg_A: tuple[int, int] = (1, 1)):
    """
    Short task description

    Args:
        arg_A: Description of arg_A.
    """
    pass


def test_tuple_argument():
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

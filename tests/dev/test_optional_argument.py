import json
from typing import Optional

from pydantic.validate_call_decorator import validate_call

from fractal_tasks_core.dev.lib_args_schemas import (
    create_schema_for_single_task,
)


@validate_call
def task_function(
    arg1: str,
    arg2: Optional[str] = None,
    arg3: Optional[list[str]] = None,
):
    """
    Short task description

    Args:
        arg1: This is the argument description
        arg2: This is the argument description
        arg3: This is the argument description
    """
    pass


def test_optional_argument():
    """
    As a first implementation of the Pydantic V2 schema generation, we are not
    supporting the `anyOf` pattern for nullable attributes. This test verifies
    that the type of nullable properties is not `anyOf`, and that they are not
    required.
    """
    schema = create_schema_for_single_task(
        task_function=validate_call(task_function),
        executable=__file__,
        package=None,
        verbose=True,
    )
    print(json.dumps(schema, indent=2, sort_keys=True))
    print()
    assert schema["properties"]["arg2"]["type"] == "string"
    assert "arg2" not in schema["required"]
    assert schema["properties"]["arg3"]["type"] == "array"
    assert "arg3" not in schema["required"]

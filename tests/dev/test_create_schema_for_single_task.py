import json

import pytest
from devtools import debug
from pydantic.decorator import validate_arguments

from fractal_tasks_core.dev.lib_args_schemas import (
    create_schema_for_single_task,
)


def test_create_schema_for_single_task_usage_1():
    """
    This test reproduces the standard schema-creation scenario, as it's found
    when running `create_manifest.py`
    """
    schema = create_schema_for_single_task(
        executable="tasks/cellpose_segmentation.py",
        package="fractal_tasks_core",
        verbose=True,
    )
    debug(schema)
    print(json.dumps(schema, indent=2))


@validate_arguments
def task_function(arg_1: int = 1):
    """
    Description

    Args:
        arg_1: Description of arg_1.
    """


def test_create_schema_for_single_task_usage_2():
    """
    This test reproduces the schema-creation scenario starting from an
    existing function, as it's done in tests.
    """
    schema = create_schema_for_single_task(
        task_function=task_function,
        executable=__file__,
        package=None,
        verbose=True,
    )
    debug(schema)
    print(json.dumps(schema, indent=2))


def test_create_schema_for_single_task_failures():
    """
    This test reproduces some invalid usage of the schema-creation function
    """
    with pytest.raises(ValueError):
        create_schema_for_single_task(
            task_function=task_function,
            executable=__file__,
            package="something",
            verbose=True,
        )
    with pytest.raises(ValueError):
        create_schema_for_single_task(
            task_function=task_function,
            executable="non_absolute/path/module.py",
            package=None,
            verbose=True,
        )
    with pytest.raises(ValueError):
        create_schema_for_single_task(
            executable="/absolute/path/cellpose_segmentation.py",
            package="fractal_tasks_core",
            verbose=True,
        )
    with pytest.raises(ValueError):
        create_schema_for_single_task(
            executable="cellpose_segmentation.py",
            package=None,
            verbose=True,
        )

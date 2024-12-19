import json

from devtools import debug

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

import json
from pathlib import Path

from devtools import debug

import fractal_tasks_core
from fractal_tasks_core.dev.lib_args_schemas import (
    create_schema_for_single_task,
)


FRACTAL_TASKS_CORE_DIR = Path(fractal_tasks_core.__file__).parent
with (FRACTAL_TASKS_CORE_DIR / "__FRACTAL_MANIFEST__.json").open("r") as f:
    MANIFEST = json.load(f)
TASK_LIST = MANIFEST["task_list"]


def test_args_title():
    """
    Check that two kinds of properties have the correct title set:
    1. Task arguments which have a custom-model type
    2. Custom-model-typed attributes of custom-model-typed task arguments

    See
    https://github.com/fractal-analytics-platform/fractal-tasks-core/issues/446
    """

    cellpose_task = next(
        task for task in TASK_LIST if task["name"] == "Cellpose Segmentation"
    )
    new_schema = create_schema_for_single_task(cellpose_task["executable"])
    properties = new_schema["properties"]
    # Standard task argument
    level_prop = properties["level"]
    debug(level_prop)
    assert level_prop["title"] == "Level"
    # Custom-model-typed task argument
    channel2_prop = properties["channel2"]
    debug(channel2_prop)
    assert channel2_prop["title"] == "Channel2"

    create_ome_zarr_task = next(
        task
        for task in TASK_LIST
        if task["name"] == "Create OME-Zarr structure"
    )
    new_schema = create_schema_for_single_task(
        create_ome_zarr_task["executable"]
    )
    definitions = new_schema["definitions"]
    omero_channel_def = definitions["OmeroChannel"]
    # Custom-model-typed attribute of custom-model-typed task argument
    window_prop = omero_channel_def["properties"]["window"]
    debug(window_prop)
    assert window_prop["title"] == "Window"

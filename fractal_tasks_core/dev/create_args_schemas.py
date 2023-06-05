"""
This script generates JSON schemas for task arguments afresh, and writes them
to the package manifest.
"""
import json
from pathlib import Path

import fractal_tasks_core
from .lib_args_schemas import create_schema_for_single_task


if __name__ == "__main__":

    # Read manifest
    manifest_path = (
        Path(fractal_tasks_core.__file__).parent / "__FRACTAL_MANIFEST__.json"
    )
    with manifest_path.open("r") as f:
        manifest = json.load(f)

    # Set or check global properties of manifest
    manifest["has_args_schemas"] = True
    manifest["args_schema_version"] = "pydantic_v1"

    # Loop over tasks and set or check args schemas
    task_list = manifest["task_list"]
    for ind, task in enumerate(task_list):
        executable = task["executable"]
        print(f"[{executable}] Start")
        try:
            schema = create_schema_for_single_task(executable)
        except AttributeError:
            print(f"[{executable}] Skip, due to AttributeError")
            print()
            continue

        manifest["task_list"][ind]["args_schema"] = schema
        print("Schema added to manifest")
        print()

    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

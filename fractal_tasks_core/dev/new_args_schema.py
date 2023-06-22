"""
Copyright 2022 (C)
    Friedrich Miescher Institute for Biomedical Research and
    University of Zurich

    Original authors:
    Tommaso Comparin <tommaso.comparin@exact-lab.it>

    This file is part of Fractal and was originally developed by eXact lab
    S.r.l.  <exact-lab.it> under contract with Liberali Lab from the Friedrich
    Miescher Institute for Biomedical Research and Pelkmans Lab from the
    University of Zurich.


Script to generate JSON schemas for task arguments afresh, and write them
to the package manifest.
"""
import json
from pathlib import Path

import fractal_tasks_core
from fractal_tasks_core.dev.lib_args_schemas import (
    create_schema_for_single_task,
)


if __name__ == "__main__":

    # Read manifest
    manifest_path = (
        Path(fractal_tasks_core.__file__).parent / "__FRACTAL_MANIFEST__.json"
    )
    with manifest_path.open("r") as f:
        manifest = json.load(f)

    # Set global properties of manifest
    manifest["has_args_schemas"] = True
    manifest["args_schema_version"] = "pydantic_v1"

    # Loop over tasks and set args schemas
    task_list = manifest["task_list"]
    for ind, task in enumerate(task_list):
        executable = task["executable"]
        print(f"[{executable}] Start")
        try:
            schema = create_schema_for_single_task(executable)
        except Exception as e:
            print(f"[{executable}] Skip. Original error:\n{str(e)}")

        manifest["task_list"][ind]["args_schema"] = schema
        print("Schema added to manifest")
        print()

    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

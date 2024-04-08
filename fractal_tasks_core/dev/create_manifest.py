# Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Tommaso Comparin <tommaso.comparin@exact-lab.it>
#
# This file is part of Fractal and was originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
"""
Script to generate JSON schemas for task arguments afresh, and write them
to the package manifest.
"""
import json
import logging
from importlib import import_module
from pathlib import Path

from fractal_tasks_core.dev.lib_args_schemas import (
    create_schema_for_single_task,
)
from fractal_tasks_core.dev.lib_task_docs import create_docs_info
from fractal_tasks_core.task_list import TASK_LIST

# from fractal_tasks_core.dev.lib_task_docs import create_docs_link


PACKAGE = "fractal_tasks_core"
MANIFEST_VERSION = "2"
HAS_ARGS_SCHEMAS = True
ARGS_SCHEMA_VERSION = "pydantic_v1"


if __name__ == "__main__":
    manifest = dict(
        manifest_version=MANIFEST_VERSION,
        task_list=[],
        has_args_schemas=HAS_ARGS_SCHEMAS,
        args_schema_version=ARGS_SCHEMA_VERSION,
    )

    for task_obj in TASK_LIST:
        task_dict = task_obj.dict(
            exclude={"meta_init", "executable_init", "meta", "executable"},
            exclude_unset=True,
        )
        if task_obj.executable_non_parallel is not None:
            task_dict[
                "executable_non_parallel"
            ] = task_obj.executable_non_parallel
        if task_obj.executable_parallel is not None:
            task_dict["executable_parallel"] = task_obj.executable_parallel

        for step in ["non_parallel", "parallel"]:
            executable = task_dict.get(f"executable_{step}")
            if executable is None:
                continue
            # Create new JSON Schema for task arguments
            logging.info(f"[{executable}] START")
            schema = create_schema_for_single_task(executable, package=PACKAGE)
            logging.info(f"[{executable}] END (new schema)")
            task_dict[f"args_schema_{step}"] = schema

        if task_obj.meta_non_parallel is not None:
            task_dict["meta_non_parallel"] = task_obj.meta_non_parallel
        if task_obj.meta_parallel is not None:
            task_dict["meta_parallel"] = task_obj.meta_parallel

        # Update docs_info, based on task-function description
        docs_info = create_docs_info(
            executable_non_parallel=task_obj.executable_non_parallel,
            executable_parallel=task_obj.executable_parallel,
        )
        # docs_link = create_docs_link(executable)
        docs_link = (
            "https://fractal-analytics-platform.github.io/fractal-tasks-core"
        )
        if docs_info:
            task_dict["docs_info"] = docs_info
        if docs_link:
            task_dict["docs_link"] = docs_link

        manifest["task_list"].append(task_dict)
        print()

    # Write manifest
    imported_package = import_module(PACKAGE)
    manifest_path = (
        Path(imported_package.__file__).parent / "__FRACTAL_MANIFEST__.json"
    )
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    logging.info(f"Up-to-date manifest stored in {manifest_path.as_posix()}")

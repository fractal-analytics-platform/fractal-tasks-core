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
from typing import Optional

from fractal_tasks_core.dev.lib_args_schemas import (
    create_schema_for_single_task,
)
from fractal_tasks_core.dev.lib_task_docs import create_docs_info


logging.basicConfig(level=logging.INFO)


ARGS_SCHEMA_VERSION = "pydantic_v2"


def create_manifest(
    package: str = "fractal_tasks_core",
    manifest_version: str = "2",
    has_args_schemas: bool = True,
    docs_link: Optional[str] = None,
    custom_pydantic_models: Optional[list[tuple[str, str, str]]] = None,
):
    """
    This function creates the package manifest based on a `task_list.py`
    Python module located in the `dev` subfolder of the package, see an
    example of such list at ...

    The manifest is then written to `__FRACTAL_MANIFEST__.json`, in the
    main `package` directory.

    Note: a valid example of `custom_pydantic_models` would be
    ```
    [
        ("my_task_package", "some_module.py", "SomeModel"),
    ]
    ```

    Arguments:
        package: The name of the package (must be importable).
        manifest_version: Only `"2"` is supported.
        has_args_schemas:
            Whether to autogenerate JSON Schemas for task arguments.
        custom_pydantic_models:
            Custom models to be included when building JSON Schemas for task
            arguments.
    """

    # Preliminary check
    if manifest_version != "2":
        raise NotImplementedError(f"{manifest_version=} is not supported")

    logging.info("Start generating a new manifest")

    # Prepare an empty manifest
    manifest = dict(
        manifest_version=manifest_version,
        task_list=[],
        has_args_schemas=has_args_schemas,
    )
    if has_args_schemas:
        manifest["args_schema_version"] = ARGS_SCHEMA_VERSION

    # Prepare a default value of docs_link
    if package == "fractal_tasks_core" and docs_link is None:
        docs_link = (
            "https://fractal-analytics-platform.github.io/fractal-tasks-core"
        )

    # Import the task list from `dev/task_list.py`
    task_list_module = import_module(f"{package}.dev.task_list")
    TASK_LIST = getattr(task_list_module, "TASK_LIST")

    # Loop over TASK_LIST, and append the proper task dictionary
    # to manifest["task_list"]
    for task_obj in TASK_LIST:
        # Convert Pydantic object to dictionary
        task_dict = task_obj.model_dump(
            exclude={"meta_init", "executable_init", "meta", "executable"},
            exclude_unset=True,
        )

        # Copy some properties from `task_obj` to `task_dict`
        if task_obj.executable_non_parallel is not None:
            task_dict[
                "executable_non_parallel"
            ] = task_obj.executable_non_parallel
        if task_obj.executable_parallel is not None:
            task_dict["executable_parallel"] = task_obj.executable_parallel
        if task_obj.meta_non_parallel is not None:
            task_dict["meta_non_parallel"] = task_obj.meta_non_parallel
        if task_obj.meta_parallel is not None:
            task_dict["meta_parallel"] = task_obj.meta_parallel

        # Autogenerate JSON Schemas for non-parallel/parallel task arguments
        if has_args_schemas:
            for kind in ["non_parallel", "parallel"]:
                executable = task_dict.get(f"executable_{kind}")
                if executable is not None:
                    logging.info(f"[{executable}] START")
                    schema = create_schema_for_single_task(
                        executable,
                        package=package,
                        custom_pydantic_models=custom_pydantic_models,
                    )
                    logging.info(f"[{executable}] END (new schema)")
                    task_dict[f"args_schema_{kind}"] = schema

        # Update docs_info, based on task-function description
        docs_info = create_docs_info(
            executable_non_parallel=task_obj.executable_non_parallel,
            executable_parallel=task_obj.executable_parallel,
            package=package,
        )
        if docs_info is not None:
            task_dict["docs_info"] = docs_info
        if docs_link is not None:
            task_dict["docs_link"] = docs_link

        manifest["task_list"].append(task_dict)
        print()

    # Write manifest
    imported_package = import_module(package)
    manifest_path = (
        Path(imported_package.__file__).parent / "__FRACTAL_MANIFEST__.json"
    )
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    logging.info(f"Manifest stored in {manifest_path.as_posix()}")


if __name__ == "__main__":
    create_manifest()

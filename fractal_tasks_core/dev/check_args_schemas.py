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

Script to check that JSON schemas for task arguments (as reported in the
package manfest) are up-to-date.
"""
import json
from pathlib import Path
from typing import Any

import fractal_tasks_core
from fractal_tasks_core.dev.lib_args_schemas import (
    create_schema_for_single_task,
)


def _compare_dicts(
    old: dict[str, Any], new: dict[str, Any], path: list[str] = []
):
    """
    Provide more informative comparison of two (possibly nested) dictionaries
    """
    path_str = "/".join(path)
    keys_old = set(old.keys())
    keys_new = set(new.keys())
    if not keys_old == keys_new:
        msg = (
            "\n\n"
            f"Dictionaries at path {path_str} have different keys:\n\n"
            f"OLD KEYS:\n{keys_old}\n\n"
            f"NEW KEYS:\n{keys_new}\n\n"
        )
        raise ValueError(msg)
    for key, value_old in old.items():
        value_new = new[key]
        if type(value_old) != type(value_new):
            msg = (
                "\n\n"
                f"Values at path {path_str}/{key} "
                "have different types:\n\n"
                f"OLD TYPE:\n{type(value_old)}\n\n"
                f"NEW TYPE:\n{type(value_new)}\n\n"
            )
            raise ValueError(msg)
        if isinstance(value_old, dict):
            _compare_dicts(value_old, value_new, path=path + [key])
        else:
            if value_old != value_new:
                msg = (
                    "\n\n"
                    f"Values at path {path_str}/{key} "
                    "are different:\n\n"
                    f"OLD VALUE:\n{value_old}\n\n"
                    f"NEW VALUE:\n{value_new}\n\n"
                )
                raise ValueError(msg)


if __name__ == "__main__":

    # Read manifest
    manifest_path = (
        Path(fractal_tasks_core.__file__).parent / "__FRACTAL_MANIFEST__.json"
    )
    with manifest_path.open("r") as f:
        manifest = json.load(f)

    # Check global properties of manifest
    if not manifest["has_args_schemas"]:
        raise ValueError(f'{manifest["has_args_schemas"]=}')
    if manifest["args_schema_version"] != "pydantic_v1":
        raise ValueError(f'{manifest["args_schema_version"]=}')

    # Loop over tasks and check args schemas
    task_list = manifest["task_list"]
    for ind, task in enumerate(task_list):

        # Read current schema
        current_schema = task["args_schema"]

        # Create new schema
        executable = task["executable"]
        print(f"[{executable}] Start")
        try:
            new_schema = create_schema_for_single_task(executable)
        except AttributeError:
            print(f"[{executable}] Skip, due to AttributeError")
            print()
            continue

        # Try to provide an informative comparison of current_schema and
        # new_schema
        _compare_dicts(current_schema, new_schema, path=[])

        # Also directly check the equality of current_schema and new_schema
        # (this is redundant, in principle)
        if current_schema != new_schema:
            raise ValueError("Schemas are different.")

        print("Schema in manifest is up-to-date.")
        print()

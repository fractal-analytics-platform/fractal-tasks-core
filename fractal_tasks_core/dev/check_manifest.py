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
Script to check that JSON schemas for task arguments (as reported in the
package manfest) are up-to-date.
"""
import json
import logging
from importlib import import_module
from pathlib import Path
from typing import Any

from fractal_tasks_core.dev.lib_args_schemas import (
    create_schema_for_single_task,
)
from fractal_tasks_core.dev.lib_task_docs import create_docs_info


PACKAGE = "fractal_tasks_core"


def _compare_dicts(
    old: dict[str, Any], new: dict[str, Any], path: list[str] = []
):
    """
    Provide more informative comparison of two (possibly nested) dictionaries.

    Args:
        old: TBD
        new: TBD
        path: TBD
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
    imported_package = import_module(PACKAGE)
    manifest_path = (
        Path(imported_package.__file__).parent / "__FRACTAL_MANIFEST__.json"
    )
    with manifest_path.open("r") as f:
        manifest = json.load(f)

    # Check global properties of manifest
    if not manifest["has_args_schemas"]:
        raise ValueError(f'{manifest["has_args_schemas"]=}')
    if manifest["args_schema_version"] != "pydantic_v2":
        raise ValueError(f'{manifest["args_schema_version"]=}')

    # Loop over tasks
    task_list = manifest["task_list"]
    for ind, task in enumerate(task_list):

        for kind in ["non_paralell", "parallel"]:
            executable = task.get(f"executable_{kind}")
            if executable is None:
                continue

            # Read current schema
            current_schema = task[f"args_schema_{kind}"]

            # Create new schema
            logging.info(f"[{executable}] START")
            new_schema = create_schema_for_single_task(
                executable, package=PACKAGE
            )

            # The following step is required because some arguments may have a
            # default which has a non-JSON type (e.g. a tuple), which we need
            # to convert to JSON type (i.e. an array) before comparison.
            new_schema = json.loads(json.dumps(new_schema))

            # Try to provide an informative comparison of current_schema and
            # new_schema
            _compare_dicts(current_schema, new_schema, path=[])

            # Also directly check the equality of current_schema and new_schema
            # (this is redundant, in principle)
            if current_schema != new_schema:
                raise ValueError("Schemas are different.")

            logging.info(
                f"[{executable}] END (task schemas are up-to-date in manifest)"
            )
            print()

        # Check docs_info and docs_link
        docs_info = create_docs_info(
            executable_non_parallel=task.get("executable_non_parallel"),
            executable_parallel=task.get("executable_parallel"),
            package=PACKAGE,
        )
        if docs_info != task.get("docs_info", ""):
            raise ValueError("docs_info not up-to-date")

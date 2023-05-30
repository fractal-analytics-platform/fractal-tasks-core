"""
This script generates JSON schemas for task arguments afresh, and writes them
to files starting like `fractal_tasks_core/__args__create_ome_zarr__.json`
"""
import argparse
import json
from importlib import import_module
from pathlib import Path
from typing import Any

from pydantic.decorator import ALT_V_ARGS
from pydantic.decorator import ALT_V_KWARGS
from pydantic.decorator import V_DUPLICATE_KWARGS
from pydantic.decorator import V_POSITIONAL_ONLY_NAME
from pydantic.decorator import ValidatedFunction

import fractal_tasks_core


FRACTAL_TASKS_CORE_DIR = Path(fractal_tasks_core.__file__).parent


def _clean_up_pydantic_generated_schema(old_schema: dict[str, Any]):
    """
    FIXME: duplicate of the function in tests/test_valid_args_schemas.py

    Strip some properties from the generated JSON Schema, see
    https://github.com/pydantic/pydantic/blob/1.10.X-fixes/pydantic/decorator.py.
    """
    new_schema = old_schema.copy()

    # Check that args and kwargs properties match with some expected dummy
    # values, and remove them from the the schema properties.
    args_property = new_schema["properties"].pop("args")
    kwargs_property = new_schema["properties"].pop("kwargs")
    expected_args_property = {"title": "Args", "type": "array", "items": {}}
    expected_kwargs_property = {"title": "Kwargs", "type": "object"}
    if args_property != expected_args_property:
        raise ValueError(
            f"{args_property=}\ndiffers from\n{expected_args_property=}"
        )
    if kwargs_property != expected_kwargs_property:
        raise ValueError(
            f"{kwargs_property=}\ndiffers from\n"
            f"{expected_kwargs_property=}"
        )

    # Remove other properties, since they come from pydantic internals
    for key in (
        V_POSITIONAL_ONLY_NAME,
        V_DUPLICATE_KWARGS,
        ALT_V_ARGS,
        ALT_V_KWARGS,
    ):
        new_schema["properties"].pop(key, None)
    return new_schema


def get_task_list_from_manifest() -> list[dict]:
    with (FRACTAL_TASKS_CORE_DIR / "__FRACTAL_MANIFEST__.json").open("r") as f:
        manifest = json.load(f)
    task_list = manifest["task_list"]
    return task_list


def create_schema_for_single_task(task: dict):
    executable = task["executable"]
    if not executable.endswith(".py"):
        raise ValueError(f"Invalid {executable=}")
    module_name = executable[:-3]
    module = import_module(f"fractal_tasks_core.{module_name}")
    task_function = getattr(module, module_name)
    vf = ValidatedFunction(task_function, config=None)
    schema = vf.model.schema()
    schema = _clean_up_pydantic_generated_schema(schema)
    return schema, module_name


if __name__ == "__main__":

    parser_main = argparse.ArgumentParser(
        description="Create/update task-arguments JSON schemas"
    )
    subparsers_main = parser_main.add_subparsers(
        title="Commands:", dest="command", required=True
    )
    parser_check = subparsers_main.add_parser(
        "check",
        description="Check that existing files are up-to-date",
        allow_abbrev=False,
    )
    parser_check = subparsers_main.add_parser(
        "new",
        description="Write new JSON schemas to files",
        allow_abbrev=False,
    )

    args = parser_main.parse_args()
    command = args.command

    with (FRACTAL_TASKS_CORE_DIR / "__FRACTAL_MANIFEST__.json").open("r") as f:
        manifest = json.load(f)
    task_list = manifest["task_list"]
    manifest["has_args_schemas"] = True
    manifest["args_schema_version"] = "pydantic_v1"

    for ind, task in enumerate(task_list):
        print(f"Now handling {task['executable']}")
        try:
            schema, module_name = create_schema_for_single_task(task)
        except AttributeError:
            print(f"Skip {module_name}, due to AttributeError")
            print()
            continue

        if command == "check":
            current_schema = task["args_schema"]
            if not current_schema == schema:
                raise ValueError("Schemas are different.")
            print("Schema in manifest is up-to-date.")
            print()
        elif command == "new":
            manifest["task_list"][ind]["args_schema"] = schema
            print("Schema added to manifest")
            print()

    with (FRACTAL_TASKS_CORE_DIR / "__FRACTAL_MANIFEST__.json").open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

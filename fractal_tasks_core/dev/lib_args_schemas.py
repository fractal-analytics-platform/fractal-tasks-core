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

Helper functions to handle JSON schemas for task arguments.
"""
import ast
from importlib import import_module
from pathlib import Path
from typing import Any

from docstring_parser import parse
from pydantic.decorator import ALT_V_ARGS
from pydantic.decorator import ALT_V_KWARGS
from pydantic.decorator import V_DUPLICATE_KWARGS
from pydantic.decorator import V_POSITIONAL_ONLY_NAME
from pydantic.decorator import ValidatedFunction

import fractal_tasks_core


_Schema = dict[str, Any]


def _remove_args_kwargs_properties(old_schema: _Schema) -> _Schema:
    """
    Remove ``args`` and ``kwargs`` schema properties

    Pydantic v1 automatically includes ``args`` and ``kwargs`` properties in
    JSON Schemas generated via ``ValidatedFunction(task_function,
    config=None).model.schema()``, with some default (empty) values -- see see
    https://github.com/pydantic/pydantic/blob/1.10.X-fixes/pydantic/decorator.py.

    Verify that these properties match with their expected default values, and
    then remove them from the schema.
    """
    new_schema = old_schema.copy()
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
    return new_schema


def _remove_pydantic_internals(old_schema: _Schema) -> _Schema:
    """
    Remove schema properties that are only used internally by Pydantic V1.
    """
    new_schema = old_schema.copy()
    for key in (
        V_POSITIONAL_ONLY_NAME,
        V_DUPLICATE_KWARGS,
        ALT_V_ARGS,
        ALT_V_KWARGS,
    ):
        new_schema["properties"].pop(key, None)
    return new_schema


def _get_args_descriptions(executable) -> dict[str, str]:
    """
    Extract argument descriptions for a task function
    """
    # Read docstring (via ast)
    module_path = Path(fractal_tasks_core.__file__).parent / executable
    module_name = module_path.with_suffix("").name
    tree = ast.parse(module_path.read_text())
    function = next(
        f
        for f in ast.walk(tree)
        if (isinstance(f, ast.FunctionDef) and f.name == module_name)
    )
    docstring = ast.get_docstring(function)
    # Parse docstring (via docstring_parser) and prepare output
    parsed_docstring = parse(docstring)
    descriptions = {
        param.arg_name: param.description.replace("\n", " ")
        for param in parsed_docstring.params
    }
    return descriptions


def _include_args_descriptions_in_schema(*, schema, descriptions):
    """
    Merge the descriptions obtained via `_get_args_descriptions` into an
    existing JSON Schema for task arguments
    """
    new_schema = schema.copy()
    new_properties = schema["properties"].copy()
    for key, value in schema["properties"].items():
        if "description" in value:
            raise ValueError("Property already has description")
        else:
            if key in descriptions:
                value["description"] = descriptions[key]
            else:
                value["description"] = "Missing description"
            new_properties[key] = value
    new_schema["properties"] = new_properties
    return new_schema


def create_schema_for_single_task(executable: str) -> _Schema:
    """
    Main function to create a JSON Schema of task arguments
    """
    if not executable.endswith(".py"):
        raise ValueError(f"Invalid {executable=} (it must end with `.py`).")
    # Import function
    module_name = Path(executable).with_suffix("").name
    module = import_module(f"fractal_tasks_core.tasks.{module_name}")
    task_function = getattr(module, module_name)

    # Create and clean up schema
    vf = ValidatedFunction(task_function, config=None)
    schema = vf.model.schema()
    schema = _remove_args_kwargs_properties(schema)
    schema = _remove_pydantic_internals(schema)

    # Include arg descriptions
    descriptions = _get_args_descriptions(executable)
    schema = _include_args_descriptions_in_schema(
        schema=schema, descriptions=descriptions
    )

    return schema

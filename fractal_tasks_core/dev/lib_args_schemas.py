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
from typing import Any

from pydantic.decorator import ALT_V_ARGS
from pydantic.decorator import ALT_V_KWARGS
from pydantic.decorator import V_DUPLICATE_KWARGS
from pydantic.decorator import V_POSITIONAL_ONLY_NAME
from pydantic.decorator import ValidatedFunction

from fractal_tasks_core.dev.lib_descriptions import _get_args_descriptions
from fractal_tasks_core.dev.lib_descriptions import (
    _include_args_descriptions_in_schema,
)
from fractal_tasks_core.dev.lib_signature_constraints import _extract_function
from fractal_tasks_core.dev.lib_signature_constraints import (
    _validate_function_signature,
)


_Schema = dict[str, Any]


INNER_PYDANTIC_MODELS = {
    "OmeroChannel": "lib_channels.py",
    "Window": "lib_channels.py",
    "Channel": "lib_input_models.py",
    "NapariWorkflowsInput": "lib_input_models.py",
    "NapariWorkflowsOutput": "lib_input_models.py",
}


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


def create_schema_for_single_task(
    executable: str,
    package: str = "fractal_tasks_core.tasks",
) -> _Schema:
    """
    Main function to create a JSON Schema of task arguments
    """
    # Extract function from module
    task_function = _extract_function(executable=executable, package=package)

    # Validate function signature against some custom constraints
    _validate_function_signature(task_function)

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

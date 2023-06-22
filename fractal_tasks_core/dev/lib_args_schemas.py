"""
Copyright 2022 (C)
    Friedrich Miescher Institute for Biomedical Research and
    University of Zurich

    Original authors:
    Tommaso Comparin <tommaso.comparin@exact-lab.it>
    Yuri Chiucconi <yuri.chiucconi@exact-lab.it>

    This file is part of Fractal and was originally developed by eXact lab
    S.r.l.  <exact-lab.it> under contract with Liberali Lab from the Friedrich
    Miescher Institute for Biomedical Research and Pelkmans Lab from the
    University of Zurich.

Helper functions to handle JSON schemas for task arguments.
"""
from pathlib import Path
from typing import Any

from pydantic.decorator import ALT_V_ARGS
from pydantic.decorator import ALT_V_KWARGS
from pydantic.decorator import V_DUPLICATE_KWARGS
from pydantic.decorator import V_POSITIONAL_ONLY_NAME
from pydantic.decorator import ValidatedFunction

from fractal_tasks_core.dev.lib_descriptions import (
    _get_class_attrs_descriptions,
)
from fractal_tasks_core.dev.lib_descriptions import (
    _get_function_args_descriptions,
)
from fractal_tasks_core.dev.lib_descriptions import (
    _insert_class_attrs_descriptions,
)
from fractal_tasks_core.dev.lib_descriptions import (
    _insert_function_args_descriptions,
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
    package: str = "fractal_tasks_core",
    inner_pydantic_models: dict[str, str] = INNER_PYDANTIC_MODELS,
) -> _Schema:
    """
    Main function to create a JSON Schema of task arguments
    """

    # Extract the function name. Note: this could be made more general, but for
    # the moment we assume the function has the same name as the module)
    function_name = Path(executable).with_suffix("").name

    # Extract function from module
    task_function = _extract_function(
        package_name=package,
        module_relative_path=executable,
        function_name=function_name,
    )

    # Validate function signature against some custom constraints
    _validate_function_signature(task_function)

    # Create and clean up schema
    vf = ValidatedFunction(task_function, config=None)
    schema = vf.model.schema()
    schema = _remove_args_kwargs_properties(schema)
    schema = _remove_pydantic_internals(schema)

    # Include arg descriptions
    function_args_descriptions = _get_function_args_descriptions(
        package_name=package,
        module_relative_path=executable,
        function_name=function_name,
    )
    schema = _insert_function_args_descriptions(
        schema=schema, descriptions=function_args_descriptions
    )

    # Include inner Pydantic models attrs descriprions
    for _class, module in INNER_PYDANTIC_MODELS.items():
        descriptions = _get_class_attrs_descriptions(
            package_name=package,
            module_relative_path=module,
            class_name=_class,
        )
        schema = _insert_class_attrs_descriptions(
            schema=schema, class_name=_class, descriptions=descriptions
        )

    return schema

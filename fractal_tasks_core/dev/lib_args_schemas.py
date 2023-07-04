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
from typing import Optional

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
from fractal_tasks_core.dev.lib_titles import _include_titles


_Schema = dict[str, Any]

FRACTAL_TASKS_CORE_PYDANTIC_MODELS = [
    ("fractal_tasks_core", "lib_channels.py", "OmeroChannel"),
    ("fractal_tasks_core", "lib_channels.py", "Window"),
    ("fractal_tasks_core", "lib_input_models.py", "Channel"),
    ("fractal_tasks_core", "lib_input_models.py", "NapariWorkflowsInput"),
    ("fractal_tasks_core", "lib_input_models.py", "NapariWorkflowsOutput"),
]


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
    custom_pydantic_models: Optional[list[tuple[str, str, str]]] = None,
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

    # Include titles for custom-model-typed arguments
    schema = _include_titles(schema)

    # Include descriptions of function arguments
    function_args_descriptions = _get_function_args_descriptions(
        package_name=package,
        module_relative_path=executable,
        function_name=function_name,
    )
    schema = _insert_function_args_descriptions(
        schema=schema, descriptions=function_args_descriptions
    )

    # Merge lists of fractal-tasks-core and user-provided Pydantic models
    user_provided_models = custom_pydantic_models or []
    pydantic_models = FRACTAL_TASKS_CORE_PYDANTIC_MODELS + user_provided_models

    # Check that model names are unique
    tmp_class_names = set()
    duplicate_class_names = set(
        item[2]
        for item in pydantic_models
        if (item[2] in tmp_class_names or tmp_class_names.add(item[2]))
    )
    if duplicate_class_names:
        pydantic_models_str = "  " + "\n  ".join(map(str, pydantic_models))
        raise ValueError(
            "Cannot parse docstrings for models with non-unique names "
            f"{duplicate_class_names}, in\n{pydantic_models_str}"
        )

    # Extract model-attribute descriptions and insert them into schema
    for package_name, module_relative_path, class_name in pydantic_models:
        attrs_descriptions = _get_class_attrs_descriptions(
            package_name=package_name,
            module_relative_path=module_relative_path,
            class_name=class_name,
        )
        schema = _insert_class_attrs_descriptions(
            schema=schema,
            class_name=class_name,
            descriptions=attrs_descriptions,
        )

    return schema

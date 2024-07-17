# Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Tommaso Comparin <tommaso.comparin@exact-lab.it>
# Yuri Chiucconi <yuri.chiucconi@exact-lab.it>
#
# This file is part of Fractal and was originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
"""
Helper functions to handle JSON schemas for task arguments.
"""
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Optional

from docstring_parser import parse as docparse
from pydantic.v1.decorator import ALT_V_ARGS
from pydantic.v1.decorator import ALT_V_KWARGS
from pydantic.v1.decorator import V_DUPLICATE_KWARGS
from pydantic.v1.decorator import V_POSITIONAL_ONLY_NAME
from pydantic.v1.decorator import ValidatedFunction

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
    ("fractal_tasks_core", "channels.py", "OmeroChannel"),
    ("fractal_tasks_core", "channels.py", "Window"),
    ("fractal_tasks_core", "channels.py", "ChannelInputModel"),
    (
        "fractal_tasks_core",
        "tasks/io_models.py",
        "NapariWorkflowsInput",
    ),
    (
        "fractal_tasks_core",
        "tasks/io_models.py",
        "NapariWorkflowsOutput",
    ),
    (
        "fractal_tasks_core",
        "tasks/cellpose_utils.py",
        "CellposeCustomNormalizer",
    ),
    (
        "fractal_tasks_core",
        "tasks/cellpose_utils.py",
        "CellposeChannel1InputModel",
    ),
    (
        "fractal_tasks_core",
        "tasks/cellpose_utils.py",
        "CellposeChannel2InputModel",
    ),
    (
        "fractal_tasks_core",
        "tasks/cellpose_utils.py",
        "CellposeModelParams",
    ),
    (
        "fractal_tasks_core",
        "tasks/io_models.py",
        "MultiplexingAcquisition",
    ),
]


def _remove_args_kwargs_properties(old_schema: _Schema) -> _Schema:
    """
    Remove `args` and `kwargs` schema properties.

    Pydantic v1 automatically includes `args` and `kwargs` properties in
    JSON Schemas generated via `ValidatedFunction(task_function,
    config=None).model.schema()`, with some default (empty) values -- see see
    https://github.com/pydantic/pydantic/blob/1.10.X-fixes/pydantic/decorator.py.

    Verify that these properties match with their expected default values, and
    then remove them from the schema.

    Args:
        old_schema: TBD
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
    logging.info("[_remove_args_kwargs_properties] END")
    return new_schema


def _remove_pydantic_internals(old_schema: _Schema) -> _Schema:
    """
    Remove schema properties that are only used internally by Pydantic V1.

    Args:
        old_schema: TBD
    """
    new_schema = old_schema.copy()
    for key in (
        V_POSITIONAL_ONLY_NAME,
        V_DUPLICATE_KWARGS,
        ALT_V_ARGS,
        ALT_V_KWARGS,
    ):
        new_schema["properties"].pop(key, None)
    logging.info("[_remove_pydantic_internals] END")
    return new_schema


def _remove_attributes_from_descriptions(old_schema: _Schema) -> _Schema:
    """
    Keeps only the description part of the docstrings: e.g from
    ```
    'Custom class for Omero-channel window, based on OME-NGFF v0.4.\\n'
    '\\n'
    'Attributes:\\n'
    'min: Do not change. It will be set to `0` by default.\\n'
    'max: Do not change. It will be set according to bitdepth of the images\\n'
    '    by default (e.g. 65535 for 16 bit images).\\n'
    'start: Lower-bound rescaling value for visualization.\\n'
    'end: Upper-bound rescaling value for visualization.'
    ```
    to `'Custom class for Omero-channel window, based on OME-NGFF v0.4.\\n'`.

    Args:
        old_schema: TBD
    """
    new_schema = old_schema.copy()
    if "definitions" in new_schema:
        for name, definition in new_schema["definitions"].items():
            parsed_docstring = docparse(definition["description"])
            new_schema["definitions"][name][
                "description"
            ] = parsed_docstring.short_description
    logging.info("[_remove_attributes_from_descriptions] END")
    return new_schema


def create_schema_for_single_task_pydantic_v1(
    executable: str,
    package: Optional[str] = "fractal_tasks_core",
    custom_pydantic_models: Optional[list[tuple[str, str, str]]] = None,
    task_function: Optional[Callable] = None,
    verbose: bool = False,
) -> _Schema:
    """
    Main function to create a JSON Schema of task arguments

    This function can be used in two ways:

    1. `task_function` argument is `None`, `package` is set, and `executable`
        is a path relative to that package.
    2. `task_function` argument is provided, `executable` is an absolute path
        to the function module, and `package` is `None. This is useful for
        testing.
    """

    DEFINITIONS_KEY = "definitions"

    logging.info("[create_schema_for_single_task] START")
    if task_function is None:
        usage = "1"
        # Usage 1 (standard)
        if package is None:
            raise ValueError(
                "Cannot call `create_schema_for_single_task with "
                f"{task_function=} and {package=}. Exit."
            )
        if os.path.isabs(executable):
            raise ValueError(
                "Cannot call `create_schema_for_single_task with "
                f"{task_function=} and absolute {executable=}. Exit."
            )
    else:
        usage = "2"
        # Usage 2 (testing)
        if package is not None:
            raise ValueError(
                "Cannot call `create_schema_for_single_task with "
                f"{task_function=} and non-None {package=}. Exit."
            )
        if not os.path.isabs(executable):
            raise ValueError(
                "Cannot call `create_schema_for_single_task with "
                f"{task_function=} and non-absolute {executable=}. Exit."
            )

    # Extract function from module
    if usage == "1":
        # Extract the function name (for the moment we assume the function has
        # the same name as the module)
        function_name = Path(executable).with_suffix("").name
        # Extract the function object
        task_function = _extract_function(
            package_name=package,
            module_relative_path=executable,
            function_name=function_name,
            verbose=verbose,
        )
    else:
        # The function object is already available, extract its name
        function_name = task_function.__name__

    if verbose:
        logging.info(f"[create_schema_for_single_task] {function_name=}")
        logging.info(f"[create_schema_for_single_task] {task_function=}")

    # Validate function signature against some custom constraints
    _validate_function_signature(task_function)

    # Create and clean up schema
    vf = ValidatedFunction(task_function, config=None)
    schema = vf.model.schema()
    schema = _remove_args_kwargs_properties(schema)
    schema = _remove_pydantic_internals(schema)
    schema = _remove_attributes_from_descriptions(schema)

    # Include titles for custom-model-typed arguments
    schema = _include_titles(
        schema, definitions_key=DEFINITIONS_KEY, verbose=verbose
    )

    # Include descriptions of function. Note: this function works both
    # for usages 1 or 2 (see docstring).
    function_args_descriptions = _get_function_args_descriptions(
        package_name=package,
        module_path=executable,
        function_name=function_name,
        verbose=verbose,
    )
    schema = _insert_function_args_descriptions(
        schema=schema, descriptions=function_args_descriptions, verbose=verbose
    )

    # Merge lists of fractal-tasks-core and user-provided Pydantic models
    user_provided_models = custom_pydantic_models or []
    pydantic_models = FRACTAL_TASKS_CORE_PYDANTIC_MODELS + user_provided_models

    # Check that model names are unique
    pydantic_models_names = [item[2] for item in pydantic_models]
    duplicate_class_names = [
        name
        for name, count in Counter(pydantic_models_names).items()
        if count > 1
    ]
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
            definition_key=DEFINITIONS_KEY,
        )

    logging.info("[create_schema_for_single_task] END")
    return schema

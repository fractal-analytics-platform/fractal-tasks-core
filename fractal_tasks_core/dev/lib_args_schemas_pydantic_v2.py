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
from copy import deepcopy
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Optional

from devtools import debug
from docstring_parser import parse as docparse
from pydantic._internal import _generate_schema
from pydantic._internal import _typing_extra
from pydantic._internal._config import ConfigWrapper
from pydantic.json_schema import GenerateJsonSchema
from pydantic.json_schema import JsonSchemaValue
from pydantic_core.core_schema import WithDefaultSchema

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
    if "$defs" in new_schema:
        for name, definition in new_schema["$defs"].items():
            parsed_docstring = docparse(definition["description"])
            new_schema["$defs"][name][
                "description"
            ] = parsed_docstring.short_description
    logging.info("[_remove_attributes_from_descriptions] END")
    return new_schema


class GenerateJsonSchemaA(GenerateJsonSchema):
    def nullable_schema(self, schema):
        null_schema = {"type": "null"}
        inner_json_schema = self.generate_inner(schema["schema"])
        if inner_json_schema == null_schema:
            return null_schema
        else:
            debug("A: Skip calling `get_flattened_anyof` method")
            return inner_json_schema


class GenerateJsonSchemaB(GenerateJsonSchemaA):
    def default_schema(self, schema: WithDefaultSchema) -> JsonSchemaValue:
        original_json_schema = super().default_schema(schema)
        new_json_schema = deepcopy(original_json_schema)
        default = new_json_schema.get("default", None)
        if default is None:
            debug("B: Pop None default")
            new_json_schema.pop("default")
        return new_json_schema


class GenerateJsonSchemaC(GenerateJsonSchema):
    def get_flattened_anyof(
        self, schemas: list[JsonSchemaValue]
    ) -> JsonSchemaValue:
        # Inspired by
        # https://github.com/vitalik/django-ninja/issues/842#issuecomment-2059014537
        original_json_schema_value = super().get_flattened_anyof(schemas)
        members = original_json_schema_value.get("anyOf")
        debug("C", original_json_schema_value)
        if (
            members is not None
            and len(members) == 2
            and {"type": "null"} in members
        ):
            new_json_schema_value = {"type": [t["type"] for t in members]}
            debug("C", new_json_schema_value)
            return new_json_schema_value
        else:
            return original_json_schema_value


class GenerateJsonSchemaD(GenerateJsonSchema):
    def get_flattened_anyof(
        self, schemas: list[JsonSchemaValue]
    ) -> JsonSchemaValue:
        # Inspired by
        # https://github.com/vitalik/django-ninja/issues/842#issuecomment-2059014537
        null_schema = {"type": "null"}
        if null_schema in schemas:
            debug("D drop null_schema before calling `get_flattened_anyof`")
            schemas.pop(schemas.index(null_schema))
        return super().get_flattened_anyof(schemas)


class GenerateJsonSchemaE(GenerateJsonSchemaD):
    def default_schema(self, schema: WithDefaultSchema) -> JsonSchemaValue:
        json_schema = super().default_schema(schema)
        debug("E", json_schema)
        if "default" in json_schema.keys() and json_schema["default"] is None:
            debug("E: Pop None default")
            json_schema.pop("default")
        return json_schema


CustomGenerateJsonSchema = GenerateJsonSchema
CustomGenerateJsonSchema = GenerateJsonSchemaA
CustomGenerateJsonSchema = GenerateJsonSchemaB
CustomGenerateJsonSchema = GenerateJsonSchemaC
CustomGenerateJsonSchema = GenerateJsonSchemaD
CustomGenerateJsonSchema = GenerateJsonSchemaE


def _create_schema_for_function(function: Callable) -> _Schema:
    namespace = _typing_extra.add_module_globals(function, None)
    gen_core_schema = _generate_schema.GenerateSchema(
        ConfigWrapper(None), namespace
    )
    core_schema = gen_core_schema.generate_schema(function)
    clean_core_schema = gen_core_schema.clean_schema(core_schema)
    gen_json_schema = CustomGenerateJsonSchema()
    json_schema = gen_json_schema.generate(
        clean_core_schema, mode="validation"
    )
    return json_schema


def create_schema_for_single_task_pydantic_v2(
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

    DEFINITIONS_KEY = "$defs"

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
    schema = _create_schema_for_function(task_function)
    schema = _remove_attributes_from_descriptions(schema)

    # Include titles for custom-model-typed arguments
    schema = _include_titles(
        schema, definitions_key=DEFINITIONS_KEY, verbose=verbose
    )

    # Include main title
    if schema.get("title") is None:

        def to_camel_case(snake_str):
            return "".join(
                x.capitalize() for x in snake_str.lower().split("_")
            )

        schema["title"] = to_camel_case(task_function.__name__)

    # Include descriptions of function. Note: this function works both
    # for usages 1 or 2 (see docstring).
    function_args_descriptions = _get_function_args_descriptions(
        package_name=package,
        module_path=executable,
        function_name=function_name,
        verbose=verbose,
    )

    schema = _insert_function_args_descriptions(
        schema=schema, descriptions=function_args_descriptions
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

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
import inspect
import logging
from importlib import import_module
from inspect import signature
from pathlib import Path
from typing import Callable

from pydantic.v1.decorator import ALT_V_ARGS
from pydantic.v1.decorator import ALT_V_KWARGS
from pydantic.v1.decorator import V_DUPLICATE_KWARGS
from pydantic.v1.decorator import V_POSITIONAL_ONLY_NAME

FORBIDDEN_PARAM_NAMES = (
    "args",
    "kwargs",
    V_POSITIONAL_ONLY_NAME,
    V_DUPLICATE_KWARGS,
    ALT_V_ARGS,
    ALT_V_KWARGS,
)


def _extract_function(
    module_relative_path: str,
    function_name: str,
    package_name: str = "fractal_tasks_core",
    verbose: bool = False,
) -> Callable:
    """
    Extract function from a module with the same name.

    Args:
        package_name: Example `fractal_tasks_core`.
        module_relative_path: Example `tasks/create_ome_zarr.py`.
        function_name: Example `create_ome_zarr`.
        verbose:
    """
    if not module_relative_path.endswith(".py"):
        raise ValueError(f"{module_relative_path=} must end with '.py'")
    module_relative_path_no_py = str(
        Path(module_relative_path).with_suffix("")
    )
    module_relative_path_dots = module_relative_path_no_py.replace("/", ".")
    if verbose:
        logging.info(
            f"Now calling `import_module` for "
            f"{package_name}.{module_relative_path_dots}"
        )
    imported_module = import_module(
        f"{package_name}.{module_relative_path_dots}"
    )
    if verbose:
        logging.info(
            f"Now getting attribute {function_name} from "
            f"imported module {imported_module}."
        )
    task_function = getattr(imported_module, function_name)
    return task_function


def _validate_function_signature(function: Callable):
    """
    Validate the function signature.

    Implement a set of checks for type hints that do not play well with the
    creation of JSON Schema, see
    https://github.com/fractal-analytics-platform/fractal-tasks-core/issues/399.

    Args:
        function: TBD
    """
    sig = signature(function)
    for param in sig.parameters.values():

        # CASE 1: Check that name is not forbidden
        if param.name in FORBIDDEN_PARAM_NAMES:
            raise ValueError(
                f"Function {function} has argument with name {param.name}"
            )

        # CASE 2: Raise an error for unions
        if str(param.annotation).startswith(("typing.Union[", "Union[")):
            raise ValueError("typing.Union is not supported")

        # CASE 3: Raise an error for "|"
        if "|" in str(param.annotation):
            raise ValueError('Use of "|" in type hints is not supported')

        # CASE 4: Raise an error for optional parameter with given (non-None)
        # default, e.g. Optional[str] = "asd"
        is_annotation_optional = str(param.annotation).startswith(
            ("typing.Optional[", "Optional[")
        )
        default_given = (param.default is not None) and (
            param.default != inspect._empty
        )
        if default_given and is_annotation_optional:
            raise ValueError("Optional parameter has non-None default value")

    logging.info("[_validate_function_signature] END")
    return sig

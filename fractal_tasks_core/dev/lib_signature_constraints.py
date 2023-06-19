import inspect
from importlib import import_module
from inspect import signature
from pathlib import Path
from typing import Callable

from pydantic.decorator import ALT_V_ARGS
from pydantic.decorator import ALT_V_KWARGS
from pydantic.decorator import V_DUPLICATE_KWARGS
from pydantic.decorator import V_POSITIONAL_ONLY_NAME

FORBIDDEN_PARAM_NAMES = (
    "args",
    "kwargs",
    V_POSITIONAL_ONLY_NAME,
    V_DUPLICATE_KWARGS,
    ALT_V_ARGS,
    ALT_V_KWARGS,
)


def _extract_function(
    executable: str,
    package: str = "fractal_tasks_core.tasks",
) -> Callable:
    """
    Extract function from a module with the same name

    This for instance extracts the function `my_function` from the module
    `my_function.py`.

    :param executable: Path to Python task script. Valid examples:
                       `tasks/my_function.py` or `my_function.py`.
    :param package: Name of the tasks subpackage (e.g.
                    `fractal_tasks_core.tasks`).
    """
    if not executable.endswith(".py"):
        raise ValueError(f"{executable=} must end with '.py'")
    module_name = Path(executable).with_suffix("").name
    module = import_module(f"{package}.{module_name}")
    task_function = getattr(module, module_name)
    return task_function


def _validate_function_signature(function: Callable):
    """
    Validate the function signature

    Implement a set of checks for type hints that do not play well with the
    creation of JSON Schema, see
    https://github.com/fractal-analytics-platform/fractal-tasks-core/issues/399.
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

    return sig

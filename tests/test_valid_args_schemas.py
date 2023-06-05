import json
from importlib import import_module
from inspect import signature
from pathlib import Path
from typing import Callable

import pytest
from devtools import debug
from jsonschema.validators import Draft201909Validator
from jsonschema.validators import Draft202012Validator
from jsonschema.validators import Draft7Validator
from pydantic.decorator import ALT_V_ARGS
from pydantic.decorator import ALT_V_KWARGS
from pydantic.decorator import V_DUPLICATE_KWARGS
from pydantic.decorator import V_POSITIONAL_ONLY_NAME

import fractal_tasks_core
from fractal_tasks_core.dev.lib_args_schemas import (
    create_schema_for_single_task,
)


FRACTAL_TASKS_CORE_DIR = Path(fractal_tasks_core.__file__).parent
with (FRACTAL_TASKS_CORE_DIR / "__FRACTAL_MANIFEST__.json").open("r") as f:
    MANIFEST = json.load(f)
TASK_LIST = MANIFEST["task_list"]

FORBIDDEN_PARAM_NAMES = (
    "args",
    "kwargs",
    V_POSITIONAL_ONLY_NAME,
    V_DUPLICATE_KWARGS,
    ALT_V_ARGS,
    ALT_V_KWARGS,
)


def _extract_function(executable: str):
    if not executable.endswith(".py"):
        raise ValueError(f"Invalid {executable=}")
    module_name = executable[:-3]
    module = import_module(f"fractal_tasks_core.{module_name}")
    task_function = getattr(module, module_name)
    return task_function


def _validate_function_signature(function: Callable):
    """
    Check that function parameters do not have forbidden names
    """
    for param in signature(function).parameters.values():
        if param.name in FORBIDDEN_PARAM_NAMES:
            raise ValueError(
                f"Function {function} has argument with name {param.name}"
            )


def test_validate_function_signature():
    """
    Showcase the expected behavior of _validate_function_signature
    """

    def fun1(x: int):
        pass

    _validate_function_signature(fun1)

    def fun2(x, *args):
        pass

    with pytest.raises(ValueError):
        _validate_function_signature(fun2)

    def fun3(x, **kwargs):
        pass

    with pytest.raises(ValueError):
        _validate_function_signature(fun3)


def test_manifest_has_args_schemas_is_true():
    debug(MANIFEST)
    assert MANIFEST["has_args_schemas"]


def test_task_functions_have_no_args_or_kwargs():
    """
    Test that task functions do not use forbidden parameter names (e.g. `args`
    or `kwargs`)
    """
    for ind_task, task in enumerate(TASK_LIST):
        task_function = _extract_function(task["executable"])
        _validate_function_signature(task_function)


def test_args_schemas_are_up_to_date():
    """
    Test that args_schema attributes in the manifest are up-to-date
    """
    for ind_task, task in enumerate(TASK_LIST):
        print(f"Now handling {task['executable']}")
        old_schema = TASK_LIST[ind_task]["args_schema"]
        new_schema = create_schema_for_single_task(task["executable"])
        assert new_schema == old_schema


@pytest.mark.parametrize(
    "jsonschema_validator",
    [Draft7Validator, Draft201909Validator, Draft202012Validator],
)
def test_args_schema_comply_with_jsonschema_specs(jsonschema_validator):
    """
    FIXME: it is not clear whether this test is actually useful
    """
    for ind_task, task in enumerate(TASK_LIST):
        schema = TASK_LIST[ind_task]["args_schema"]
        my_validator = jsonschema_validator(schema=schema)
        my_validator.check_schema(my_validator.schema)
        print(
            f"Schema for task {task['executable']} is valid for "
            f"{jsonschema_validator}."
        )

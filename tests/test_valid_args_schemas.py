import json
from importlib import import_module
from pathlib import Path

import pytest
from devtools import debug
from jsonschema.validators import Draft201909Validator
from jsonschema.validators import Draft202012Validator
from jsonschema.validators import Draft7Validator

import fractal_tasks_core


FRACTAL_TASKS_CORE_DIR = Path(fractal_tasks_core.__file__).parent
with (FRACTAL_TASKS_CORE_DIR / "__FRACTAL_MANIFEST__.json").open("r") as f:
    MANIFEST = json.load(f)
TASK_LIST = MANIFEST["task_list"]


def _create_schema_for_single_task(task: dict):
    executable = task["executable"]
    if not executable.endswith(".py"):
        raise ValueError(f"Invalid {executable=}")
    module_name = executable[:-3]
    module = import_module(f"fractal_tasks_core.{module_name}")
    TaskArguments = getattr(module, "TaskArguments")
    schema = TaskArguments.schema()
    return schema


def _extract_function(task: dict):
    executable = task["executable"]
    if not executable.endswith(".py"):
        raise ValueError(f"Invalid {executable=}")
    module_name = executable[:-3]
    module = import_module(f"fractal_tasks_core.{module_name}")
    task_function = getattr(module, module_name)
    return task_function


def test_manifest_has_args_schemas_is_true():
    debug(MANIFEST)
    assert MANIFEST["has_args_schemas"]


def test_args_schemas_are_up_to_date():
    for ind_task, task in enumerate(TASK_LIST):
        print(f"Now handling {task['executable']}")
        new_schema = _create_schema_for_single_task(task)
        old_schema = TASK_LIST[ind_task]["args_schema"]
        if not new_schema == old_schema:
            raise ValueError("Schemas are different.")
        print(f"Schema for task {task['executable']} is up-to-date.")
        print()


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
        print()


def test_args_schema_match_with_function_arguments():
    for ind_task, task in enumerate(TASK_LIST):
        print(f"Now handling {task['executable']}")
        schema = _create_schema_for_single_task(task)
        fun = _extract_function(task)
        debug(fun)
        names_from_signature = set(
            name
            for name, _type in fun.__annotations__.items()
            if name != "return"
        )
        name_from_args_schema = set(schema["properties"].keys())
        assert names_from_signature == name_from_args_schema

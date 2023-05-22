import json
from importlib import import_module
from pathlib import Path

import fractal_tasks_core


FRACTAL_TASKS_CORE_DIR = Path(fractal_tasks_core.__file__).parent


def get_task_list_from_manifest() -> list[dict]:
    with (FRACTAL_TASKS_CORE_DIR / "__FRACTAL_MANIFEST__.json").open("r") as f:
        manifest = json.load(f)
    task_list = manifest["task_list"]
    return task_list


def create_schema_for_single_task(task: dict):
    executable = task["executable"]
    if not executable.endswith(".py"):
        raise ValueError(f"Invalid {executable=}")
    module_name = executable[:-3]
    module = import_module(f"fractal_tasks_core.{module_name}")
    TaskArguments = getattr(module, "TaskArguments")
    schema = TaskArguments.schema()
    return schema, module_name


def test_task_arguments_schemas():

    task_list = get_task_list_from_manifest()
    for task in task_list:
        print(f"Now handling {task['executable']}")
        try:
            schema, module_name = create_schema_for_single_task(task)
        except AttributeError as e:
            print(f"Skip {module_name}, due to AttributeError")
            print()
            raise e

        schema_path = FRACTAL_TASKS_CORE_DIR / f"__args__{module_name}__.json"
        with schema_path.open("r") as f:
            current_schema = json.load(f)
        if not current_schema == schema:
            raise ValueError("Schemas are different.")
        print(f"Schema in {schema_path.as_posix()} is up-to-date.")
        print()

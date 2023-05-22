"""
This script generates JSON schemas for task arguments afresh, and writes them
to files starting like `fractal_tasks_core/__args__create_ome_zarr__.json`
"""
import json
from importlib import import_module
from pathlib import Path

import fractal_tasks_core


# Load manifest
module_dir = Path(fractal_tasks_core.__file__).parent
with (module_dir / "__FRACTAL_MANIFEST__.json").open("r") as f:
    manifest = json.load(f)

for task in manifest["task_list"]:
    executable = task["executable"]
    if not executable.endswith(".py"):
        raise ValueError(f"Invalid {executable=}")
    module_name = executable[:-3]
    print(f"Now handling {module_name}")
    module = import_module(f"fractal_tasks_core.{module_name}")
    try:
        TaskArguments = getattr(module, "TaskArguments")
    except AttributeError:
        print(f"Skip {module_name}, due to AttributeError")
        print()
        continue
    schema = TaskArguments.schema()
    schema_path = module_dir / f"__args__{module_name}__.json"
    with schema_path.open("w") as f:
        json.dump(schema, f, indent=2)
    print(f"Schema written to {schema_path.as_posix()}")
    print()

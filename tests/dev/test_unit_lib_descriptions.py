import json

from devtools import debug

from fractal_tasks_core.dev.lib_descriptions import (
    _get_class_attrs_descriptions,
)
from fractal_tasks_core.dev.lib_descriptions import (
    _get_function_args_descriptions,
)
from fractal_tasks_core.dev.lib_descriptions import (
    _insert_class_attrs_descriptions,
)

# from fractal_tasks_core.dev.lib_descriptions import (
#     _insert_function_args_descriptions,
# )


def test_get_function_args_descriptions():
    args_descriptions = _get_function_args_descriptions(
        "fractal_tasks_core",
        "dev/lib_signature_constraints.py",
        "_extract_function",
    )
    debug(args_descriptions)
    assert args_descriptions.keys() == set(("executable", "package"))


def test_get_class_attrs_descriptions():
    attrs_descriptions = _get_class_attrs_descriptions(
        "fractal_tasks_core", "lib_input_models.py", "Channel"
    )
    debug(attrs_descriptions)
    assert attrs_descriptions.keys() == set(("wavelength_id", "label"))


def test_descriptions():
    FILE = "lib_channels.py"
    CLASS = "OmeroChannel"

    descriptions = _get_class_attrs_descriptions(
        "fractal_tasks_core", FILE, CLASS
    )

    with open("fractal_tasks_core/__FRACTAL_MANIFEST__.json", "r") as f:
        manifest = json.load(f)

    schemas = [task["args_schema"] for task in manifest["task_list"]]

    for _, schema in enumerate(schemas):
        new_schema = _insert_class_attrs_descriptions(
            schema=schema, class_name=CLASS, descriptions=descriptions
        )
        if "definitions" in schema:
            for class_name, definition in new_schema["definitions"].items():
                if class_name == "OmeroChannel":
                    for prop in definition["properties"]:
                        assert "description" in definition["properties"][prop]

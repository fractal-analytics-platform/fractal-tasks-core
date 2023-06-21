import json

import pytest
from devtools import debug

from fractal_tasks_core.dev.lib_descriptions import (
    _get_attributes_models_descriptions,
)
from fractal_tasks_core.dev.lib_descriptions import (
    _get_function_args_descriptions,
)
from fractal_tasks_core.dev.lib_descriptions import (
    _include_attributs_descriptions_in_schema,
)


def test_get_args_model_descriptions():

    x = _get_function_args_descriptions(
        "fractal_tasks_core", "lib_input_models.py", "Channel"
    )
    debug(x, dir(x))
    # assert False

    with pytest.raises(ValueError):
        _get_attributes_models_descriptions(models={"Foo": "__init__.py"})

    descriptions = _get_attributes_models_descriptions()
    debug(descriptions)

    with open("fractal_tasks_core/__FRACTAL_MANIFEST__.json", "r") as f:
        manifest = json.load(f)

    schemas = [task["args_schema"] for task in manifest["task_list"]]

    for i, schema in enumerate(schemas):
        new_schema = _include_attributs_descriptions_in_schema(
            schema=schema, descriptions=descriptions
        )
        if "definitions" in schema:
            for _, definition in new_schema["definitions"].items():
                for prop in definition["properties"]:
                    assert "description" in definition["properties"][prop]

    with pytest.raises(ValueError):
        _include_attributs_descriptions_in_schema(
            schema=new_schema, descriptions=descriptions
        )

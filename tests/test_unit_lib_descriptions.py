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


def test_get_function_args_descriptions():
    function_doc = _get_function_args_descriptions(
        "fractal_tasks_core",
        "dev.lib_signature_constraints.py",
        "_extract_function",
    )
    assert function_doc.keys() == set(("executable", "package"))


def test_descriptions():

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

import ast
from pathlib import Path

from docstring_parser import parse as docparse

import fractal_tasks_core

inner_pydantic_models = {
    "OmeroChannel": "lib_channels.py",
    "Window": "lib_channels.py",
    "Channel": "tasks/_input_models.py",
    "NapariWorkflowsInput": "tasks/_input_models.py",
    "NapariWorkflowsOutput": "tasks/_input_models.py",
}


def _get_args_model_descriptions():

    descriptions = {}

    for model, module in inner_pydantic_models.items():

        module_path = Path(fractal_tasks_core.__file__).parent / module
        tree = ast.parse(module_path.read_text())
        _class = next(
            c
            for c in ast.walk(tree)
            if (isinstance(c, ast.ClassDef) and c.name == model)
        )
        if not _class:
            raise RuntimeError(f"Model {module_path}::{model} not found.")
        else:
            descriptions[model] = {}

        docstring = ast.get_docstring(_class)
        if docstring:
            descriptions[model]["__docstring__"] = docstring

        var_name: str = ""
        for node in _class.body:
            if isinstance(node, ast.AnnAssign):
                descriptions[model][node.target.id] = None
                var_name = node.target.id
            else:
                if isinstance(node, ast.Expr) and var_name:
                    descriptions[model][var_name] = node.value.s
                    var_name = ""

    return descriptions


def _get_args_descriptions(executable) -> dict[str, str]:
    """
    Extract argument descriptions for a task function
    """
    # Read docstring (via ast)
    module_path = Path(fractal_tasks_core.__file__).parent / executable
    module_name = module_path.with_suffix("").name
    tree = ast.parse(module_path.read_text())
    function = next(
        f
        for f in ast.walk(tree)
        if (isinstance(f, ast.FunctionDef) and f.name == module_name)
    )

    docstring = ast.get_docstring(function)
    # Parse docstring (via docstring_parser) and prepare output
    parsed_docstring = docparse(docstring)
    descriptions = {
        param.arg_name: param.description.replace("\n", " ")
        for param in parsed_docstring.params
    }
    return descriptions


def _include_args_descriptions_in_schema(*, schema, descriptions):
    """
    Merge the descriptions obtained via `_get_args_descriptions` into an
    existing JSON Schema for task arguments
    """
    new_schema = schema.copy()
    new_properties = schema["properties"].copy()
    for key, value in schema["properties"].items():
        if "description" in value:
            raise ValueError("Property already has description")
        else:
            if key in descriptions:
                value["description"] = descriptions[key]
            else:
                value["description"] = "Missing description"
            new_properties[key] = value
    new_schema["properties"] = new_properties
    return new_schema

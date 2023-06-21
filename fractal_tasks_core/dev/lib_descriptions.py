import ast
from importlib import import_module
from pathlib import Path

from docstring_parser import parse as docparse

import fractal_tasks_core


def _get_function_args_descriptions(
    package_name: str, module_relative_path: str, function_name: str
) -> dict[str, str]:
    """
    Extract argument descriptions from a function
    """

    if not module_relative_path.endswith(".py"):
        raise ValueError(f"Module {module_relative_path} must end with '.py'")
    else:
        module_relative_path = module_relative_path.rstrip(".py")
    module_name = Path(package_name).with_suffix("").name
    module = import_module(f"{module_name}.{module_relative_path}")
    function = getattr(module, function_name)

    docstring = function.__doc__

    # Parse docstring (via docstring_parser) and prepare output
    parsed_docstring = docparse(docstring)
    descriptions = {
        param.arg_name: param.description.replace("\n", " ")
        for param in parsed_docstring.params
    }
    return descriptions


def _get_class_attrs_descriptions(
    package_name: str, module_relative_path: str, class_name: str
) -> dict[str, str]:
    if not module_relative_path.endswith(".py"):
        raise ValueError(f"Module {module_relative_path} must end with '.py'")

    # get the class
    module_path = Path(package_name) / module_relative_path
    tree = ast.parse(module_path.read_text())

    _class = next(
        c
        for c in ast.walk(tree)
        if (isinstance(c, ast.ClassDef) and c.name == class_name)
    )
    attrs_descriptions = {}

    # extract attribute docstrings
    var_name: str = ""
    for node in _class.body:
        if isinstance(node, ast.AnnAssign):
            attrs_descriptions[node.target.id] = "Missing description"
            var_name = node.target.id
        else:
            if isinstance(node, ast.Expr) and var_name:
                attrs_descriptions[var_name] = node.value.s
                var_name = ""
    return attrs_descriptions


def _get_args_descriptions(executable) -> dict[str, str]:
    """
    Extract argument descriptions for a task function.
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
    existing JSON Schema for task arguments.
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


INNER_PYDANTIC_MODELS = {
    "OmeroChannel": "lib_channels.py",
    "Window": "lib_channels.py",
    "Channel": "lib_input_models.py",
    "NapariWorkflowsInput": "lib_input_models.py",
    "NapariWorkflowsOutput": "lib_input_models.py",
}


def _get_attributes_models_descriptions(
    models: dict[str, str] = INNER_PYDANTIC_MODELS
) -> dict[str, dict[str, str]]:
    """
    Extract attribut descriptions for Pydantic models
    """
    descriptions = {}

    for model, module in models.items():
        # get the class
        module_path = Path(fractal_tasks_core.__file__).parent / module
        tree = ast.parse(module_path.read_text())
        try:
            _class = next(
                c
                for c in ast.walk(tree)
                if (isinstance(c, ast.ClassDef) and c.name == model)
            )
            descriptions[model] = {}
        except StopIteration:
            raise ValueError(f"Model {module_path}::{model} not found.")

        # extract attribute docstrings
        var_name: str = ""
        for node in _class.body:
            if isinstance(node, ast.AnnAssign):
                descriptions[model][node.target.id] = "Missing description"
                var_name = node.target.id
            else:
                if isinstance(node, ast.Expr) and var_name:
                    descriptions[model][var_name] = node.value.s
                    var_name = ""

    return descriptions


def _include_attributs_descriptions_in_schema(*, schema, descriptions):
    """
    Merge the descriptions obtained via `_get_attributes_models_descriptions`
    into an existing JSON Schema for task arguments.
    """
    new_schema = schema.copy()

    if "definitions" not in schema:
        return new_schema
    else:
        new_definitions = schema["definitions"].copy()

    for name, definition in schema["definitions"].items():
        if name in descriptions.keys():
            for prop in definition["properties"]:
                if "description" in new_definitions[name]["properties"][prop]:
                    raise ValueError(
                        f"Property {name}.{prop} already has description"
                    )
                else:
                    new_definitions[name]["properties"][prop][
                        "description"
                    ] = descriptions[name][prop]
    new_schema["definitions"] = new_definitions
    return new_schema

import ast
from importlib import import_module
from pathlib import Path

from docstring_parser import parse as docparse


def _sanitize_description(string: str) -> str:
    """
    Sanitize a description string.

    This is a provisional helper function that replaces newlines with spaces
    and reduces multiple contiguous whitespace characters to a single one.
    Future iterations of the docstrings format/parsing may render this function
    not-needed or obsolete.
    """
    # Replace newline with space
    new_string = string.replace("\n", " ")
    # Replace N-whitespace characterss with a single one
    while "  " in new_string:
        new_string = new_string.replace("  ", " ")
    return new_string


def _get_function_args_descriptions(
    package_name: str, module_relative_path: str, function_name: str
) -> dict[str, str]:
    """
    Extract argument descriptions from a function

    :param package_name: Example ``fractal_tasks_core``
    :param module_relative_path: Example ``tasks/create_ome_zarr.py``
    :param function_name: Example ``create_ome_zarr``
    """

    if not module_relative_path.endswith(".py"):
        raise ValueError(f"Module {module_relative_path} must end with '.py'")

    # Get the function ast.FunctionDef object
    package_path = Path(import_module(package_name).__file__).parent
    module_path = package_path / module_relative_path
    tree = ast.parse(module_path.read_text())
    _function = next(
        f
        for f in ast.walk(tree)
        if (isinstance(f, ast.FunctionDef) and f.name == function_name)
    )

    # Extract docstring from ast.FunctionDef
    docstring = ast.get_docstring(_function)

    # Parse docstring (via docstring_parser) and prepare output
    parsed_docstring = docparse(docstring)
    descriptions = {
        param.arg_name: _sanitize_description(param.description)
        for param in parsed_docstring.params
    }
    return descriptions


def _get_class_attrs_descriptions(
    package_name: str, module_relative_path: str, class_name: str
) -> dict[str, str]:
    """
    Extract attribute descriptions from a class

    :param package_name: Example ``fractal_tasks_core``
    :param module_relative_path: Example ``lib_channels.py``
    :param class_name: Example ``OmeroChannel``
    """

    if not module_relative_path.endswith(".py"):
        raise ValueError(f"Module {module_relative_path} must end with '.py'")

    # Get the class ast.ClassDef object
    package_path = Path(import_module(package_name).__file__).parent
    module_path = package_path / module_relative_path
    tree = ast.parse(module_path.read_text())
    try:
        _class = next(
            c
            for c in ast.walk(tree)
            if (isinstance(c, ast.ClassDef) and c.name == class_name)
        )
    except StopIteration:
        raise RuntimeError(
            f"Cannot find {class_name=} for {package_name=} "
            f"and {module_relative_path=}"
        )

    descriptions = {}
    # extract attribute docstrings
    var_name: str = ""
    for node in _class.body:
        if isinstance(node, ast.AnnAssign):
            descriptions[node.target.id] = "Missing description"
            var_name = node.target.id
        else:
            if isinstance(node, ast.Expr) and var_name:
                descriptions[var_name] = _sanitize_description(node.value.s)
                var_name = ""
    return descriptions


def _insert_function_args_descriptions(*, schema, descriptions):
    """
    Merge the descriptions obtained via `_get_args_descriptions` into the
    properties of an existing JSON Schema.

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


def _insert_class_attrs_descriptions(*, schema, class_name, descriptions):
    """
    Merge the descriptions obtained via `_get_attributes_models_descriptions`
    into the ``class_name`` definition, within an existing JSON Schema
    """
    new_schema = schema.copy()
    if "definitions" not in schema:
        return new_schema
    else:
        new_definitions = schema["definitions"].copy()
    # Loop over existing definitions
    for name, definition in schema["definitions"].items():
        if name == class_name:
            for prop in definition["properties"]:
                if "description" in new_definitions[name]["properties"][prop]:
                    raise ValueError(
                        f"Property {name}.{prop} already has description"
                    )
                else:
                    new_definitions[name]["properties"][prop][
                        "description"
                    ] = descriptions[prop]
    new_schema["definitions"] = new_definitions
    return new_schema

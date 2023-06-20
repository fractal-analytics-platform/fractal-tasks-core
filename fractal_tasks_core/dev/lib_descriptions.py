import ast
from pathlib import Path

from docstring_parser import parse

import fractal_tasks_core


inner_pydantic_models = {
    "OmeroChannel": "fractal_tasks_core.lib_channels.py",
    "Window": "fractal_tasks_core.lib_channels.py",
    "Channel": "fractal_tasks_core.tasks._input_models.py",
    "NapariWorkflowsInput": "fractal_tasks_core.tasks._input_models.py",
    "NapariWorkflowsOutput": "fractal_tasks_core.tasks._input_models.py",
}


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
    parsed_docstring = parse(docstring)
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

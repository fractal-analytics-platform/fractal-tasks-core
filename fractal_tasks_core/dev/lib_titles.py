"""
Module to include titles in JSON Schema properties
"""
from typing import Any


_Schema = dict[str, Any]


def _include_titles_for_properties(
    properties: dict[str, dict]
) -> dict[str, dict]:
    """
    Scan through properties of a JSON Schema, and set their title when it is
    missing.

    The title is set to `name.title()`, where `title` is a standard string
    method - see https://docs.python.org/3/library/stdtypes.html#str.title.
    """
    new_properties = properties.copy()
    for prop_name, prop in properties.items():
        if "title" not in prop.keys():
            new_prop = prop.copy()
            new_prop["title"] = prop_name.title()
            new_properties[prop_name] = new_prop
    return new_properties


def _include_titles(schema: _Schema) -> _Schema:
    """
    Include property titles, when missing

    This handles both (1) first-level JSON Schema properties (corresponding to
    task arguments) and (2) properties of JSON Schema definitions
    (corresponding to task-argument attributes).
    """
    new_schema = schema.copy()

    # Update first-level properties (that is, task arguments)
    new_properties = _include_titles_for_properties(schema["properties"])
    new_schema["properties"] = new_properties

    # Update properties of definitions
    if "definitions" in schema.keys():
        new_definitions = schema["definitions"].copy()
        for def_name, def_schema in new_definitions.items():
            new_properties = _include_titles_for_properties(
                def_schema["properties"]
            )
            new_definitions[def_name]["properties"] = new_properties
        new_schema["definitions"] = new_definitions

    return new_schema

# Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Tommaso Comparin <tommaso.comparin@exact-lab.it>
#
# This file is part of Fractal and was originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
"""
Module to include titles in JSON Schema properties.
"""
import logging
from typing import Any


_Schema = dict[str, Any]


def _include_titles_for_properties(
    properties: dict[str, dict],
    verbose: bool = False,
) -> dict[str, dict]:
    """
    Scan through properties of a JSON Schema, and set their title when it is
    missing.

    The title is set to `name.title()`, where `title` is a standard string
    method - see https://docs.python.org/3/library/stdtypes.html#str.title.

    Args:
        properties: TBD
    """
    if verbose:
        logging.info(
            f"[_include_titles_for_properties] Original properties:\n"
            f"{properties}"
        )

    new_properties = properties.copy()
    for prop_name, prop in properties.items():
        if "title" not in prop.keys():
            new_prop = prop.copy()
            new_prop["title"] = prop_name.title()
            new_properties[prop_name] = new_prop
    if verbose:
        logging.info(
            f"[_include_titles_for_properties] New properties:\n"
            f"{new_properties}"
        )
    return new_properties


def _include_titles(
    schema: _Schema,
    definitions_key: str,
    verbose: bool = False,
) -> _Schema:
    """
    Include property titles, when missing.

    This handles both:

    - first-level JSON Schema properties (corresponding to task
        arguments);
    - properties of JSON Schema definitions (corresponding to
        task-argument attributes).

    Args:
        schema: TBD
        definitions_key: Either `"definitions"` (for Pydantic V1) or
            `"$defs"` (for Pydantic V2)
        verbose:
    """
    new_schema = schema.copy()

    if verbose:
        logging.info("[_include_titles] START")
        logging.info(f"[_include_titles] Input schema:\n{schema}")

    # Update first-level properties (that is, task arguments)
    new_properties = _include_titles_for_properties(
        schema["properties"], verbose=verbose
    )
    new_schema["properties"] = new_properties

    if verbose:
        logging.info("[_include_titles] Titles for properties now included.")

    # Update properties of definitions
    if definitions_key in schema.keys():
        new_definitions = schema[definitions_key].copy()
        for def_name, def_schema in new_definitions.items():
            if "properties" not in def_schema.keys():
                if verbose:
                    logging.info(
                        f"Definition schema {def_name} has no 'properties' "
                        "key. Skip."
                    )
            else:
                new_properties = _include_titles_for_properties(
                    def_schema["properties"], verbose=verbose
                )
                new_definitions[def_name]["properties"] = new_properties
        new_schema[definitions_key] = new_definitions

    if verbose:
        logging.info(
            "[_include_titles] Titles for definitions properties now included."
        )
        logging.info("[_include_titles] END")
    return new_schema

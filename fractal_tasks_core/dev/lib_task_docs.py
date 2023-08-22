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
import logging
from pathlib import Path

from docstring_parser import parse as docparse

from fractal_tasks_core.dev.lib_descriptions import _get_function_docstring


def _get_function_description(
    package_name: str, module_relative_path: str, function_name: str
) -> str:
    """
    Extract function description from its docstring.

    Args:
        package_name: Example `fractal_tasks_core`.
        module_relative_path: Example `tasks/create_ome_zarr.py`.
        function_name: Example `create_ome_zarr`.
    """
    # Extract docstring from ast.FunctionDef
    docstring = _get_function_docstring(
        package_name, module_relative_path, function_name
    )
    # Parse docstring (via docstring_parser)
    parsed_docstring = docparse(docstring)
    # Combine short/long descriptions (if present)
    short_description = parsed_docstring.short_description
    long_description = parsed_docstring.long_description
    items = []
    if short_description:
        items.append(short_description)
    if long_description:
        items.append(long_description)
    if items:
        if parsed_docstring.blank_after_short_description:
            return "\n\n".join(items)
        else:
            return "\n".join(items)
    else:
        return ""


def create_docs_info(
    executable: str,
    package: str = "fractal_tasks_core",
) -> str:
    """
    Return task description based on function docstring.
    """
    logging.info("[create_docs_info] START")
    # Extract the function name. Note: this could be made more general, but for
    # the moment we assume the function has the same name as the module)
    function_name = Path(executable).with_suffix("").name
    logging.info(f"[create_docs_info] {function_name=}")
    # Get function description
    docs_info = _get_function_description(
        package_name=package,
        module_relative_path=executable,
        function_name=function_name,
    )
    logging.info("[create_docs_info] END")
    return docs_info


def create_docs_link(executable: str) -> str:
    """
    Return link to docs page for a fractal_tasks_core task.
    """
    logging.info("[create_docs_link] START")

    # Extract the function name. Note: this could be made more general, but for
    # the moment we assume the function has the same name as the module)
    function_name = Path(executable).with_suffix("").name
    logging.info(f"[create_docs_link] {function_name=}")
    # Define docs_link
    docs_link = (
        "https://fractal-analytics-platform.github.io/fractal-tasks-core/"
        f"reference/fractal_tasks_core/tasks/{function_name}/"
        f"#fractal_tasks_core.tasks.{function_name}.{function_name}"
    )
    logging.info("[create_docs_link] END")
    return docs_link

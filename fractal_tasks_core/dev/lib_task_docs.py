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
from typing import Optional

from docstring_parser import parse as docparse

from fractal_tasks_core.dev.lib_descriptions import _get_function_docstring


def _get_function_description(
    package_name: str, module_path: str, function_name: str
) -> str:
    """
    Extract function description from its docstring.

    Args:
        package_name: Example `fractal_tasks_core`.
        module_path: Example `tasks/create_ome_zarr.py`.
        function_name: Example `create_ome_zarr`.
    """
    # Extract docstring from ast.FunctionDef
    docstring = _get_function_docstring(
        package_name=package_name,
        module_path=module_path,
        function_name=function_name,
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
    executable_non_parallel: Optional[str] = None,
    executable_parallel: Optional[str] = None,
    package: str = "fractal_tasks_core",
) -> list[str]:
    """
    Return task description based on function docstring.
    """
    logging.info("[create_docs_info] START")
    docs_info = []
    for executable in [executable_non_parallel, executable_parallel]:
        if executable is None:
            continue
        # Extract the function name.
        # Note: this could be made more general, but for the moment we assume
        # that the function has the same name as the module)
        function_name = Path(executable).with_suffix("").name
        logging.info(f"[create_docs_info] {function_name=}")
        # Get function description
        description = _get_function_description(
            package_name=package,
            module_path=executable,
            function_name=function_name,
        )
        docs_info.append(f"## {function_name}\n{description}\n")
    docs_info = "".join(docs_info)
    logging.info("[create_docs_info] END")
    return docs_info

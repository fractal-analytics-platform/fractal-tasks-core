# Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Tommaso Comparin <tommaso.comparin@exact-lab.it>
# Yuri Chiucconi <yuri.chiucconi@exact-lab.it>
#
# This file is part of Fractal and was originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
"""
Helper functions to handle JSON schemas for task arguments.
"""
from typing import Any
from typing import Callable
from typing import Literal
from typing import Optional


_Schema = dict[str, Any]


def create_schema_for_single_task(
    executable: str,
    package: Optional[str] = "fractal_tasks_core",
    custom_pydantic_models: Optional[list[tuple[str, str, str]]] = None,
    task_function: Optional[Callable] = None,
    verbose: bool = False,
    args_schema_version: Literal["pydantic_v1", "pydantic_v2"] = "pydantic_v1",
) -> _Schema:
    """
    Wrapper of pydantic v1/v2 functions
    """

    if args_schema_version == "pydantic_v1":
        from fractal_tasks_core.dev.lib_args_schemas_pydantic_v1 import (
            create_schema_for_single_task_pydantic_v1,
        )

        print("START WITH PYDANTIC V1")
        schema = create_schema_for_single_task_pydantic_v1(
            executable=executable,
            package=package,
            custom_pydantic_models=custom_pydantic_models,
            task_function=task_function,
            verbose=verbose,
        )
    elif args_schema_version == "pydantic_v2":
        from fractal_tasks_core.dev.lib_args_schemas_pydantic_v2 import (
            create_schema_for_single_task_pydantic_v2,
        )

        print("START WITH PYDANTIC V2")

        schema = create_schema_for_single_task_pydantic_v2(
            executable=executable,
            package=package,
            custom_pydantic_models=custom_pydantic_models,
            task_function=task_function,
            verbose=verbose,
        )
    return schema

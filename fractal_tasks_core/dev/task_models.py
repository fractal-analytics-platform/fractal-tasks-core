# Copyright 2024 (C) Friedrich Miescher Institute for Biomedical Research and
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
Task models for Fractal tasks.

These models are used in `task_list.py`, and they provide a layer that
simplifies writing the task list of a package in a way that is compliant with
fractal-server v2.
"""
from typing import Any
from typing import Optional

from pydantic import BaseModel


class _BaseTask(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"

    name: str
    executable: str
    meta: Optional[dict[str, Any]] = None
    input_types: Optional[dict[str, bool]] = None
    output_types: Optional[dict[str, bool]] = None


class CompoundTask(_BaseTask):
    """
    A `CompoundTask` object must include both `executable_init` and
    `executable` attributes, and it may include the `meta_init` and `meta`
    attributes.
    """

    executable_init: str
    meta_init: Optional[dict[str, Any]] = None

    @property
    def executable_non_parallel(self) -> str:
        return self.executable_init

    @property
    def meta_non_parallel(self) -> Optional[dict[str, Any]]:
        return self.meta_init

    @property
    def executable_parallel(self) -> str:
        return self.executable

    @property
    def meta_parallel(self) -> Optional[dict[str, Any]]:
        return self.meta


class NonParallelTask(_BaseTask):
    """
    A `NonParallelTask` object must include the `executable` attribute, and it
    may include the `meta` attribute.
    """

    @property
    def executable_non_parallel(self) -> str:
        return self.executable

    @property
    def meta_non_parallel(self) -> Optional[dict[str, Any]]:
        return self.meta

    @property
    def executable_parallel(self) -> None:
        return None

    @property
    def meta_parallel(self) -> None:
        return None


class ParallelTask(_BaseTask):
    """
    A `ParallelTask` object must include the `executable` attribute, and it may
    include the `meta` attribute.
    """

    @property
    def executable_non_parallel(self) -> None:
        return None

    @property
    def meta_non_parallel(self) -> None:
        return None

    @property
    def executable_parallel(self) -> str:
        return self.executable

    @property
    def meta_parallel(self) -> Optional[dict[str, Any]]:
        return self.meta

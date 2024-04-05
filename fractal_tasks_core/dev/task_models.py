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
    meta: Optional[dict[str, Any]]
    input_types: Optional[dict[str, bool]]
    output_types: Optional[dict[str, bool]]


class CompoundTask(_BaseTask):
    executable_init: str
    meta_init: Optional[dict[str, Any]]

    @property
    def executable_non_parallel(self) -> str:
        return self.executable_init

    @property
    def executable_parallel(self) -> str:
        return self.executable

    @property
    def meta_non_parallel(self) -> Optional[dict[str, Any]]:
        return self.meta_init

    @property
    def meta_parallel(self) -> Optional[dict[str, Any]]:
        return self.meta


class ParallelTask(_BaseTask):
    @property
    def executable_non_parallel(self) -> None:
        return None

    @property
    def executable_parallel(self) -> str:
        return self.executable

    @property
    def meta_non_parallel(self) -> None:
        return None

    @property
    def meta_parallel(self) -> Optional[dict[str, Any]]:
        return self.meta


class NonParallelTask(_BaseTask):
    @property
    def executable_parallel(self) -> None:
        return None

    @property
    def executable_non_parallel(self) -> str:
        return self.executable

    @property
    def meta_parallel(self) -> None:
        return None

    @property
    def meta_non_parallel(self) -> Optional[dict[str, Any]]:
        return self.meta

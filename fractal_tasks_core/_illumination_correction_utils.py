# Copyright 2022-2026 (C) BioVisionCenter, University of Zurich
"""Utility models for the illumination_correction task."""

from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from pydantic import BaseModel


class ProfileCorrectionModel(BaseModel):
    """Parameters for profile-based corrections."""

    model_config = {"title": "Correction with Profiles"}
    model: Literal["Profile"] = "Profile"
    """
    The correction model to be applied.
    """
    folder: str
    """
    Path of folder of correction profiles.
    """
    profiles: dict[str, str]
    """
    Dictionary where keys match the "wavelength_id" attributes of existing channels
    (e.g. "A01_C01") and values are the filenames of the corresponding
    correction profiles.
    """

    def items(
        self,
    ) -> Iterator[tuple[str, str],]:
        root_path = Path(self.folder)
        for wavelength_id, profile_name in self.profiles.items():
            yield wavelength_id, (root_path / profile_name).as_posix()


class ConstantCorrectionModel(BaseModel):
    """Parameters for constant-based corrections."""

    model_config = {"title": "Correction with constants"}
    model: Literal["Constant"] = "Constant"
    """
    The correction model to be applied.
    """
    constants: dict[str, int]
    """
    Dictionary where keys match the "wavelength_id" attributes of existing channels
    (e.g. "A01_C01") and values are the constant values to be used for correction.
    """


class NoCorrectionModel(BaseModel):
    """Select for no correction to be applied."""

    model_config = {"title": "No Correction"}
    model: Literal["No Correction"] = "No Correction"
    """
    The correction model to be applied.
    """

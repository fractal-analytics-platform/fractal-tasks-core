# Copyright 2022-2026 (C) BioVisionCenter, University of Zurich
"""Utility models for the illumination_correction task."""

from collections.abc import Iterator
from pathlib import Path
from typing import Literal, Tuple

from pydantic import BaseModel


class ProfileCorrectionModel(BaseModel):
    """
    Parameters for profile-based corrections.

    Attributes:
        model: The correction model to be applied.
        folder: Path of folder of correction profiles.
        profiles: Dictionary where keys match the `wavelength_id`
            attributes of existing channels (e.g.  `A01_C01` ) and values are
            the filenames of the corresponding correction profiles.
    """

    model_config = {"title": "Correction with Profiles"}
    model: Literal["Profile"] = "Profile"
    folder: str
    profiles: dict[str, str]

    def items(
        self,
    ) -> Iterator[Tuple[str, str],]:
        root_path = Path(self.folder)
        for wavelength_id, profile_name in self.profiles.items():
            yield wavelength_id, (root_path / profile_name).as_posix()


class ConstantCorrectionModel(BaseModel):
    """
    Parameters for constant-based corrections.

    Attributes:
        model: The correction model to be applied.
        constants: Dictionary where keys match the `wavelength_id`
            attributes of existing channels (e.g.  `A01_C01` ) and values are
            the constant values to be used for correction.
    """

    model_config = {"title": "Correction with constants"}
    model: Literal["Constant"] = "Constant"
    constants: dict[str, int]


class NoCorrectionModel(BaseModel):
    """
    Select for no correction to be applied.

    Attributes:
        model: The correction model to be applied.

    """

    model_config = {"title": "No Correction"}
    model: Literal["No Correction"] = "No Correction"

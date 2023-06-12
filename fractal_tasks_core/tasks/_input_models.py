"""
Copyright 2022 (C)
    Friedrich Miescher Institute for Biomedical Research and
    University of Zurich

    Original authors:
    Tommaso Comparin <tommaso.comparin@exact-lab.it>

    This file is part of Fractal and was originally developed by eXact lab
    S.r.l.  <exact-lab.it> under contract with Liberali Lab from the Friedrich
    Miescher Institute for Biomedical Research and Pelkmans Lab from the
    University of Zurich.

Pydantic models for some task parameters
"""
from typing import Optional

from pydantic import BaseModel
from pydantic import validator


class BaseChannel(BaseModel):
    """
    TBD
    """

    wavelength_id: Optional[str] = None
    label: Optional[str] = None

    @validator("label", always=True)
    def mutually_exclusive_channel_attributes(cls, v, values):
        """
        If `label` is set, then `wavelength_id` must be `None`
        """
        wavelength_id = values.get("wavelength_id")
        label = v

        if wavelength_id is not None and v is not None:
            raise ValueError(
                "`wavelength_id` and `label` cannot be both set "
                f"(given {wavelength_id=} and {label=})."
            )

        if wavelength_id is None and v is None:
            raise ValueError(
                "`wavelength_id` and `label` cannot be both `None`"
            )

        return v


class CellposeSegmentationChannel(BaseChannel):
    """
    TBD
    """

    pass


class NapariWorkflowsChannel(BaseModel):
    """
    TBD
    """

    pass

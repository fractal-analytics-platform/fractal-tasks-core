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
from typing import Literal
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


class NapariWorkflowsInputSpecsItem(BaseModel):
    """
    TBD
    """

    type: Literal["image", "label"]
    channel: BaseChannel


class NapariWorkflowsOutputSpecsItem(BaseModel):
    """
    TBD
    """

    type: Literal["label", "dataframe"]
    channel: BaseChannel
    label_name: Optional[str] = None
    table_name: Optional[str] = None

    @validator("label_name", always=True)
    def label_name_only_for_label_type(cls, v, values):
        if v and values.get("type") != "label":
            raise ValueError("FIXME")

    @validator("table_name", always=True)
    def table_name_only_for_dataframe_type(cls, v, values):
        if v and values.get("type") != "dataframe":
            raise ValueError("FIXME")

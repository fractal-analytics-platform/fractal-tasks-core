from typing import Literal
from typing import Optional

from pydantic import BaseModel
from pydantic import validator

from fractal_tasks_core.channels import ChannelInputModel


class NapariWorkflowsOutput(BaseModel):
    """
    A value of the `output_specs` argument in `napari_workflows_wrapper`.

    Attributes:
        type: Output type (either `label` or `dataframe`).
        label_name: Label name (for label outputs, it is used as the name of
            the label; for dataframe outputs, it is used to fill the
            `region["path"]` field).
        table_name: Table name (for dataframe outputs only).
    """

    type: Literal["label", "dataframe"]
    label_name: str
    table_name: Optional[str] = None

    @validator("table_name", always=True)
    def table_name_only_for_dataframe_type(cls, v, values):
        """
        Check that table_name is set only for dataframe outputs.
        """
        _type = values.get("type")
        if (_type == "dataframe" and (not v)) or (_type != "dataframe" and v):
            raise ValueError(
                f"Output item has type={_type} but table_name={v}."
            )
        return v


class NapariWorkflowsInput(BaseModel):
    """
    A value of the `input_specs` argument in `napari_workflows_wrapper`.

    Attributes:
        type: Input type (either `image` or `label`).
        label_name: Label name (for label inputs only).
        channel: `ChannelInputModel` object (for image inputs only).
    """

    type: Literal["image", "label"]
    label_name: Optional[str]
    channel: Optional[ChannelInputModel]

    @validator("label_name", always=True)
    def label_name_is_present(cls, v, values):
        """
        Check that label inputs have `label_name` set.
        """
        _type = values.get("type")
        if _type == "label" and not v:
            raise ValueError(
                f"Input item has type={_type} but label_name={v}."
            )
        return v

    @validator("channel", always=True)
    def channel_is_present(cls, v, values):
        """
        Check that image inputs have `channel` set.
        """
        _type = values.get("type")
        if _type == "image" and not v:
            raise ValueError(f"Input item has type={_type} but channel={v}.")
        return v

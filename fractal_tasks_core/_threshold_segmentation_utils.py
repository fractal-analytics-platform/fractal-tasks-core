"""Pydantic models and utilities specific to threshold segmentation."""

import logging
from typing import Annotated, Literal

import numpy as np
from ngio import ChannelSelectionModel
from pydantic import BaseModel, Field
from skimage.filters import threshold_otsu
from skimage.measure import label

logger = logging.getLogger("threshold_segmentation_task_utils")


class CreateMaskingRoiTable(BaseModel):
    """Create Masking ROI Table Configuration.

    Attributes:
        mode (Literal["Create Masking ROI Table"]): Mode to create masking ROI table.
        table_name (str): Name of the masking ROI table to be created.
            Defaults to "{label_name}_masking_ROI_table", where {label_name} is
            the name of the label image used for segmentation.
    """

    mode: Literal["Create Masking ROI Table"] = "Create Masking ROI Table"
    table_name: str = "{label_name}_masking_ROI_table"

    def get_table_name(self, label_name: str) -> str:
        """Get the actual table name by replacing placeholder.

        Args:
            label_name (str): Name of the label image used for segmentation.

        Returns:
            str: Actual name of the masking ROI table.
        """
        return self.table_name.format(label_name=label_name)


class SkipCreateMaskingRoiTable(BaseModel):
    """Skip Creating Masking ROI Table Configuration.

    Attributes:
        mode (Literal["Skip Creating Masking ROI Table"]): Mode to skip creating
            masking ROI table.
    """

    mode: Literal["Skip Creating Masking ROI Table"] = "Skip Creating Masking ROI Table"


AnyCreateRoiTableModel = Annotated[
    CreateMaskingRoiTable | SkipCreateMaskingRoiTable,
    Field(discriminator="mode"),
]


class InputChannel(BaseModel):
    """Input channel configuration for threshold segmentation.

    This model is used to select a channel by label, wavelength ID, or index.

    Attributes:
        mode (Literal["label", "wavelength_id", "index"]): Specifies how to
            interpret the identifier. Can be "label", "wavelength_id", or
            "index" (must be an integer string).
        identifier (str): Unique identifier for the channel. This can be a
            channel label, wavelength ID, or index.
        skip_if_missing (bool): If True and the specified channel is not found in
            the image, the segmentation will be skipped instead of raising an error.
            Defaults to False.
    """

    mode: Literal["label", "wavelength_id", "index"] = "label"
    identifier: str
    skip_if_missing: bool = False

    def to_channel_selection_models(self) -> ChannelSelectionModel:
        """Convert to ChannelSelectionModel.

        Returns:
            ChannelSelectionModel: Channel selection model.
        """
        return ChannelSelectionModel(identifier=self.identifier, mode=self.mode)


class ThresholdConfiguration(BaseModel):
    """Configuration for threshold-based segmentation.

    Attributes:
        method (Literal["threshold"]): Discriminator for threshold-based segmentation.
        threshold (float): Threshold value to apply for segmentation.
    """

    method: Literal["threshold"] = "threshold"
    threshold: float

    def threshold_value(self, image: np.ndarray) -> float:
        """Return the fixed threshold value."""
        return self.threshold


class OtsuConfiguration(BaseModel):
    """Configuration for Otsu threshold-based segmentation.

    Attributes:
        method (Literal["otsu"]): Discriminator for Otsu threshold-based segmentation.
    """

    method: Literal["otsu"] = "otsu"

    def threshold_value(self, image: np.ndarray) -> float:
        """Calculate Otsu threshold value for the given image."""
        return threshold_otsu(image)


SegmentationConfiguration = Annotated[
    ThresholdConfiguration | OtsuConfiguration,
    Field(discriminator="method"),
]


def segmentation_function(
    input_image: np.ndarray,
    method: SegmentationConfiguration,
) -> np.ndarray:
    """Apply threshold-based segmentation to a single image chunk.

    Pre- and post-processing transforms are handled by the segmentation iterator
    and should be configured via SegmentationTransformConfig.

    Args:
        input_image (np.ndarray): Input image data (after pre-processing transforms).
        method (SegmentationConfiguration): Configuration for the segmentation method.

    Returns:
        np.ndarray: Segmented label image with a leading channel dimension.
    """
    threshold_value = method.threshold_value(input_image)
    logger.info(f"Calculated threshold value: {threshold_value}")
    masks = input_image > threshold_value
    label_img = label(masks)
    assert isinstance(label_img, np.ndarray), "Label image must be a numpy array"
    return np.expand_dims(label_img, axis=0).astype(np.uint32)

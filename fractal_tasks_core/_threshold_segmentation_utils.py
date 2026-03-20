# Copyright 2022-2026 (C) BioVisionCenter, University of Zurich
"""Pydantic models and utilities specific to threshold segmentation."""

import logging
from typing import Annotated, Literal

import numpy as np
from ngio import ChannelSelectionModel, OmeZarrContainer
from pydantic import BaseModel, Field
from skimage.filters import threshold_otsu
from skimage.measure import label

from fractal_tasks_core._utils import AVAILABLE_TABLE_BACKENDS, DEFAULT_TABLE_BACKEND

logger = logging.getLogger("threshold_segmentation_task_utils")


class CreateMaskingRoiTable(BaseModel):
    """Create Masking ROI Table Configuration.

    Attributes:
        mode (Literal["Create Masking ROI Table"]): Mode to create masking ROI table.
        table_name (str): Name of the masking ROI table to be created.
            Defaults to "{output_label_name}_masking_ROI_table", where
            {output_label_name} is the name of the label image used for segmentation.
    """

    mode: Literal["Create Masking ROI Table"] = "Create Masking ROI Table"
    table_name: str = "{output_label_name}_masking_ROI_table"
    """
    Name of the masking ROI table to be created. This can include the placeholder
    "{output_label_name}", which will be replaced by the name of the label image used
    for segmentation.
    """
    table_backend: AVAILABLE_TABLE_BACKENDS = DEFAULT_TABLE_BACKEND
    """
    Backend to use for storing the masking ROI table. Options are "anndata", "json",
    "csv", and "parquet".
    """

    def get_table_name(self, output_label_name: str) -> str:
        """Get the actual table name by replacing placeholder.

        Args:
            output_label_name (str): Name of the label image used for segmentation.

        Returns:
            str: Actual name of the masking ROI table.
        """
        return self.table_name.format(output_label_name=output_label_name)

    def create(
        self, ome_zarr: OmeZarrContainer, output_label_name: str, overwrite: bool = True
    ) -> None:
        """Create the masking ROI table based on the provided label image.

        Args:
            ome_zarr (OmeZarrContainer): The OME-Zarr container to add the table to.
            output_label_name (str): The name of the label image for which to create
                the masking ROI table.
            overwrite (bool): Whether to overwrite an existing table. Defaults to True.
        """
        table_name = self.get_table_name(output_label_name)
        label_img = ome_zarr.get_label(name=output_label_name)
        masking_roi_table = label_img.build_masking_roi_table()
        ome_zarr.add_table(
            name=table_name,
            table=masking_roi_table,
            overwrite=overwrite,
            backend=self.table_backend,
        )


class SkipCreateMaskingRoiTable(BaseModel):
    """Skip Creating Masking ROI Table Configuration."""

    mode: Literal["Skip Creating Masking ROI Table"] = "Skip Creating Masking ROI Table"
    """
    Mode to skip creating masking ROI table.
    """

    def create(
        self, ome_zarr: OmeZarrContainer, output_label_name: str, overwrite: bool = True
    ) -> None:
        """No-op create method for skipping masking ROI table creation."""
        pass


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
    """
    Specifies how to interpret the identifier for selecting the channel. Can be
    "label" to select by channel label, "wavelength_id" to select by wavelength ID,
    or "index" to select by channel index (the identifier must be an integer string).
    """
    identifier: str
    """
    Identifier for the channel to use for segmentation.
    """
    skip_if_missing: bool = False
    """
    If True and the specified channel is not found in the image, the segmentation
    will be skipped instead of raising an error. Defaults to False.
    """

    def to_channel_selection_models(self) -> ChannelSelectionModel:
        """Convert to ChannelSelectionModel.

        Returns:
            ChannelSelectionModel: Channel selection model.
        """
        return ChannelSelectionModel(identifier=self.identifier, mode=self.mode)


class SimpleThresholdConfiguration(BaseModel):
    """Configuration for threshold-based segmentation.

    Attributes:
        method (Literal["Simple Threshold"]): Discriminator for simple
            threshold-based segmentation.
        threshold (float): Threshold value to apply for segmentation.
    """

    method: Literal["Simple Threshold"] = "Simple Threshold"
    """
    Simple threshold-based segmentation using a fixed threshold value.
    """

    threshold: float
    """
    Threshold value to use for segmentation. All pixels with intensity greater
    than this value will be considered foreground.
    """

    def threshold_value(self, image: np.ndarray) -> float:
        """Return the fixed threshold value."""
        return self.threshold


class OtsuConfiguration(BaseModel):
    """Configuration for Otsu threshold-based segmentation.

    Attributes:
        method (Literal["Otsu"]): Discriminator for Otsu threshold-based segmentation.
    """

    method: Literal["Otsu"] = "Otsu"
    """
    Otsu's method automatically determines an optimal threshold value by maximizing
    the variance between foreground and background pixel intensities.
    """

    def threshold_value(self, image: np.ndarray) -> float:
        """Calculate Otsu threshold value for the given image."""
        return threshold_otsu(image)


SegmentationConfiguration = Annotated[
    SimpleThresholdConfiguration | OtsuConfiguration,
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
    if not isinstance(label_img, np.ndarray):
        raise TypeError("Label image must be a numpy array")
    label_img = label_img.astype(np.uint32)
    return label_img

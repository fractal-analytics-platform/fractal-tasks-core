"""Pydantic models for advanced iterator configuration."""

import logging
from typing import Annotated, Literal

import numpy as np
from ngio import ChannelSelectionModel
from pydantic import BaseModel, Field
from skimage.filters import gaussian, median, threshold_otsu
from skimage.morphology import remove_small_objects

logger = logging.getLogger("threshold_segmentation_task_utils")


class MaskingConfiguration(BaseModel):
    """Masking configuration.

    Attributes:
        mode (Literal["Table Name", "Label Name"]): Mode of masking to be applied.
            If "Table Name", the identifier refers to a masking table name.
            If "Label Name", the identifier refers to a label image name.
        identifier (str | None): Name of the masking table or label image
            depending on the mode.
    """

    mode: Literal["Table Name", "Label Name"] = "Table Name"
    identifier: str | None = None


class IteratorConfiguration(BaseModel):
    """Advanced Masking configuration.

    Attributes:
        masking (MaskingConfiguration | None): If set, the segmentation will be
            performed only within the confines of the specified mask. A mask can be
            specified either by a label image or a Masking ROI table.
        roi_table (str | None): Name of a ROI table. If set, the segmentation
            will be applied to each ROI in the table individually. This option can
            be combined with masking.
    """

    masking: MaskingConfiguration | None = Field(
        default=None, title="Masking Iterator Configuration"
    )
    roi_table: str | None = Field(default=None, title="Iterate Over ROIs")


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
    """Cellpose channels configuration.

    This model is used to select a channel by label, wavelength ID, or index.

    Attributes:
        mode (Literal["label", "wavelength_id", "index"]): Specifies how to
            interpret the identifier. Can be "label", "wavelength_id", or
            "index" (must be an integer).
        identifiers (list[str]): Unique identifiers for the channels. This can
            be channel labels, wavelength IDs, or indices. At least one and at
            most three identifiers must be provided.
        skip_if_missing (bool): If True and the specified channel(s) are not found in
            the image, the segmentation will be skipped instead of raising an error.
            Defaults to False.
    """

    mode: Literal["label", "wavelength_id", "index"] = "label"
    identifier: str
    skip_if_missing: bool = False

    def to_channel_selection_models(self) -> ChannelSelectionModel:
        """Convert to list of ChannelSelectionModel.

        Returns:
            list[ChannelSelectionModel]: List of ChannelSelectionModel.
        """
        return ChannelSelectionModel(identifier=self.identifier, mode=self.mode)


class ThresholdConfiguration(BaseModel):
    """Configuration for threshold-based segmentation.

    Attributes:
        method (Literal["threshold"]): Discriminator for threshold-based segmentation.
        threshold_value (float): Threshold value to apply for segmentation.
    """

    method: Literal["threshold"] = "threshold"
    threshold: float

    def threshold_value(self, image: np.ndarray) -> float:
        """Validate that the threshold value is non-negative."""
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

### Pre and post-processing configurations for threshold-based segmentation tasks


class GaussianFilter(BaseModel):
    """Gaussian pre-processing configuration.

    Attributes:
        type (Literal["gaussian"]): Type of pre-processing.
        sigma_xy (float): Standard deviation for Gaussian kernel in XY plane.
        sigma_z (float | None): Standard deviation for Gaussian kernel in Z axis.
            If not specified, no smoothing is applied in Z axis.
    """

    type: Literal["gaussian"] = "gaussian"
    sigma_xy: float = Field(default=2.0, gt=0)
    sigma_z: float | None = None

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply Gaussian filter to the image.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Filtered image.
        """
        if image.ndim == 2:
            if self.sigma_z is not None:
                logger.warning(
                    "sigma_z is specified but the input image is 2D. Ignoring sigma_z."
                )
            return gaussian(image, sigma=self.sigma_xy)
        elif image.ndim == 3:
            sigma = (
                self.sigma_z if self.sigma_z is not None else 0,
                self.sigma_xy,
                self.sigma_xy,
            )
            return gaussian(image, sigma=sigma)  # type: ignore[call-arg] is correct
        elif image.ndim == 4:
            sigma = (
                0,
                self.sigma_z if self.sigma_z is not None else 0,
                self.sigma_xy,
                self.sigma_xy,
            )
            return gaussian(image, sigma=sigma)  # type: ignore[call-arg] is correct
        else:
            raise ValueError("Input to Gaussian filter image must be 2D, 3D, or 4D.")


class MedianFilter(BaseModel):
    """Median filter pre-processing configuration.

    Attributes:
        type (Literal["median"]): Type of pre-processing.
        size_xy (int): Size in pixels of the median filter in XY plane.
        size_z (int | None): Size in pixels of the median filter in Z axis.
            If not specified, no filtering is applied in Z axis.
    """

    type: Literal["median"] = "median"
    size_xy: int = Field(default=2, gt=0)
    size_z: int | None = None

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply Median filter to the image.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Filtered image.
        """
        if image.ndim == 2:
            if self.size_z is not None:
                logger.warning(
                    "size_z is specified but the input image is 2D. Ignoring size_z."
                )
            return median(image, footprint=np.ones((self.size_xy, self.size_xy)))
        elif image.ndim == 3:
            size = (
                self.size_z if self.size_z is not None else 1,
                self.size_xy,
                self.size_xy,
            )
            return median(image, footprint=np.ones(size))
        elif image.ndim == 4:
            size = (
                1,
                self.size_z if self.size_z is not None else 1,
                self.size_xy,
                self.size_xy,
            )
            return median(image, footprint=np.ones(size))
        else:
            raise ValueError("Input to median filter image must be 2D, 3D, or 4D.")


PreProcess = Annotated[GaussianFilter | MedianFilter, Field(discriminator="type")]


class SizeFilter(BaseModel):
    """Size filter post-processing configuration.

    Attributes:
        type (Literal["size_filter"]): Type of post-processing.
        min_size (int): Minimum size in pixels for objects to keep.
    """

    type: Literal["size_filter"] = "size_filter"
    min_size: int = Field(ge=0)

    def apply(self, labels: np.ndarray) -> np.ndarray:
        """Apply size filtering to the labeled image.

        Args:
            labels (np.ndarray): Labeled image.

        Returns:
            np.ndarray: Size-filtered labeled image.
        """
        return remove_small_objects(labels, max_size=self.min_size)


PostProcess = Annotated[
    SizeFilter,
    Field(discriminator="type"),
]


class PrePostProcessConfiguration(BaseModel):
    """Configuration for pre- and post-processing steps.

    Attributes:
        pre_process (list[PreProcess]): List of pre-processing steps.
        post_process (list[PostProcess]): List of post-processing steps.
    """

    pre_process: list[PreProcess] = Field(default_factory=list)
    post_process: list[PostProcess] = Field(default_factory=list)


def apply_pre_process(
    image: np.ndarray,
    pre_process_steps: list[PreProcess],
) -> np.ndarray:
    """Apply pre-processing steps to the image.

    Args:
        image (np.ndarray): Input image.
        pre_process_steps (list[PreProcess]): List of pre-processing steps.

    Returns:
        np.ndarray: Pre-processed image.
    """
    for step in pre_process_steps:
        image = step.apply(image)
    return image


def apply_post_process(
    labels: np.ndarray,
    post_process_steps: list[PostProcess],
) -> np.ndarray:
    """Apply post-processing steps to the labeled image.

    Args:
        labels (np.ndarray): Labeled image.
        post_process_steps (list[PostProcess]): List of post-processing steps.

    Returns:
        np.ndarray: Post-processed labeled image.
    """
    for step in post_process_steps:
        labels = step.apply(labels)
    return labels

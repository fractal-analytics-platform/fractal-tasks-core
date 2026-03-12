# Copyright 2022-2026 (C) BioVisionCenter, University of Zurich
"""Utility models and functions for the measure_features task."""

from typing import Annotated, Literal

import numpy as np
from ngio import (
    ChannelSelectionModel,
    Image,
    Roi,
)
from pydantic import BaseModel, Field
from skimage import measure

AvailableTableBackends = Literal["anndata", "json", "csv", "parquet"]


class ShapeFeatures(BaseModel):
    """Shape features extracted from regionprops."""

    include_convex_hull_properties: bool = False
    """
    Whether to include convex hull related properties like area_convex, area_filled,
    extent, solidity. These are not included since they can sometimes return
    unexpected Warnings.
    """

    type: Literal["ShapeFeatures"] = "ShapeFeatures"

    def property_names(self, is_2d: bool) -> list[str]:
        _base_properties = [
            "area",
            "area_bbox",
            "num_pixels",
            "equivalent_diameter_area",
            "axis_major_length",
            "axis_minor_length",
            "euler_number",
        ]
        _base_2d_properties = [
            "feret_diameter_max",
            "perimeter",
            "perimeter_crofton",
            "eccentricity",
            "orientation",
        ]
        _convex_hull_properties = [
            "area_convex",
            "area_filled",
            "extent",
            "solidity",
        ]
        properties: list[str] = []
        properties.extend(_base_properties)
        if is_2d:
            properties.extend(_base_2d_properties)
        if self.include_convex_hull_properties:
            properties.extend(_convex_hull_properties)

        return properties


class InputChannel(BaseModel):
    """Input channel configuration for measurement tasks.

    This model is used to select a channel by label, wavelength ID, or index.

    Attributes:
        mode (Literal["label", "wavelength_id", "index"]): Specifies how to
            interpret the identifier. Can be "label", "wavelength_id", or
            "index" (must be an integer string).
        identifier (str): Unique identifier for the channel. This can be a
            channel label, wavelength ID, or index.
    """

    mode: Literal["label", "wavelength_id", "index"] = "label"
    identifier: str

    def to_channel_selection_models(self) -> ChannelSelectionModel:
        """Convert to ChannelSelectionModel.

        Returns:
            ChannelSelectionModel: Channel selection model.
        """
        return ChannelSelectionModel(identifier=self.identifier, mode=self.mode)


class IntensityFeatures(BaseModel):
    """Intensity features extracted from regionprops."""

    type: Literal["IntensityFeatures"] = "IntensityFeatures"
    channels: list[InputChannel] | None = None
    """
    List of channels to extract intensity features from. If None all
    channels will be used.
    """

    def property_names(self, is_2d: bool) -> list[str]:
        return [
            "intensity_mean",
            "intensity_median",
            "intensity_max",
            "intensity_min",
            "intensity_std",
        ]

    def to_channel_selection_models(self) -> list[ChannelSelectionModel] | None:
        """Convert to list of ChannelSelectionModel.

        Returns:
            list[ChannelSelectionModel] | None: List of channel selection models,
                or None if no channels are specified.
        """
        if self.channels is None:
            return None
        return [channel.to_channel_selection_models() for channel in self.channels]


SupportedFeatures = Annotated[
    ShapeFeatures | IntensityFeatures,
    Field(discriminator="type"),
]


class AdvancedOptions(BaseModel):
    """Advanced options for feature measurement."""

    level_path: str | None = None
    """
    Optional path to the pyramid level to use for the measurement.
    If None, the highest resolution level will be used.
    """

    use_scaling: bool = True
    """
    Whether to use pixel scaling from the OME-Zarr metadata. This will scale the
    features according to the physical pixel size, e.g. area will be in
    square microns instead of square pixels. Defaults to True.
    """

    use_cache: bool = True
    """
    Whether to cache the loaded images. Caching can speed up the measurement
    if multiple features are extracted from the same image, but it can also
    increase memory usage. Defaults to True.
    """

    table_backend: AvailableTableBackends = "anndata"
    """
    Table backend to use for the output table. Defaults to "anndata".
    """


def _prepare_regionprops_kwargs(
    image: Image,
    list_features: list[SupportedFeatures],
    use_scaling: bool = True,
    use_cache: bool = True,
) -> tuple[dict, list[ChannelSelectionModel] | None]:
    """Prepare keyword arguments for regionprops based on the requested features.

    This includes determining which channels to load for intensity features.

    Args:
        image (Image): The image to check against.
        list_features (list[SupportedFeatures]): List of requested features.
        use_scaling (bool): Whether to use pixel scaling from the image metadata.
            Defaults to True.
        use_cache (bool): Whether to cache the loaded images. Defaults to True.
    Returns:
        tuple[dict, list[ChannelSelectionModel] | None]: Keyword arguments to pass
            to regionprops, e.g. {"intensity_image": ...}, and the list
            of channel selection models.
    """
    is_2d = image.is_2d
    kwargs = {}
    properties = ["label"]  # label is always needed for regionprops
    channel_selection_models = None
    for feature in list_features:
        properties.extend(feature.property_names(is_2d=is_2d))
        if isinstance(feature, IntensityFeatures):
            channel_selection_models = feature.to_channel_selection_models()

    if channel_selection_models is not None:
        channel_identifiers = {
            i: m.identifier for i, m in enumerate(channel_selection_models)
        }
    else:
        channel_identifiers = {i: label for i, label in enumerate(image.channel_labels)}

    px_size = image.pixel_size
    if is_2d:
        spacings = [px_size.get(ax, 1.0) for ax in "yx"]
    else:
        spacings = [px_size.get(ax, 1.0) for ax in "yxz"]

    kwargs["properties"] = properties
    kwargs["channel_identifiers"] = channel_identifiers
    kwargs["spacings"] = spacings if use_scaling else None
    kwargs["use_cache"] = use_cache
    return kwargs, channel_selection_models


def region_props_features_func(
    image: np.ndarray,
    label: np.ndarray,
    roi: Roi,
    properties: list[str],
    channel_identifiers: dict[int, str] | None = None,
    spacings: list[float] | None = None,
    use_cache: bool = True,
) -> dict:
    """Extract region properties features from a label image within a ROI."""
    if image.ndim not in (3, 4):
        raise ValueError("Image must be 3D yxc or 4D yxzc ")
    if image.ndim != label.ndim:
        raise ValueError("Image and label must have the same number of dimensions")
    # Since we query the label image as yxc or yxzc, we need to ensure
    # it has a single channel and we need to squeeze the channel dimension
    if label.shape[-1] != 1:
        raise ValueError("Label image must have a single channel")
    label = label[..., 0]

    # Perform region props extraction
    props = measure.regionprops_table(
        label_image=label,
        intensity_image=image,
        properties=properties,
        spacing=spacings,
        cache=use_cache,
    )
    # Rename channel-specific features
    if channel_identifiers is not None:
        for i, identifier in channel_identifiers.items():
            for key in list(props.keys()):
                suffix = f"-{i}"
                if key.endswith(suffix):
                    old_base, _ = key.split(suffix)
                    new_key = old_base + f"-{identifier}"
                    props[new_key] = props.pop(key)
    # Add some more metadata columns, e.g. the ROI name
    num_regions = len(props["label"])
    props["region"] = [roi.get_name()] * num_regions

    # Check if time axis is present in the ROI and add the time point as a column
    t_slice = roi.get(axis_name="t")
    if t_slice is not None and t_slice.start is not None:
        # Add the time point as a column if the ROI has a time axis
        props["time_index"] = [t_slice.start] * num_regions
    return props

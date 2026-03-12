# Copyright 2022-2026 (C) BioVisionCenter, University of Zurich
"""Measure region-properties features from OME-Zarr label images."""

import logging
from typing import Annotated, Literal

import numpy as np
from fractal_tasks_utils.measurement import (
    compute_measurement,
    setup_measurement_iterator,
)
from ngio import (
    ChannelSelectionModel,
    Image,
    Roi,
    open_ome_zarr_container,
)
from ngio.tables import FeatureTable
from pydantic import BaseModel, Field, validate_call
from skimage import measure

logger = logging.getLogger("measure_features")

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
    assert image.ndim in (3, 4), "Image must be 3D yxc or 4D yxzc "
    assert image.ndim == label.ndim, (
        "Image and label must have the same number of dimensions"
    )
    # Since we query the label image as yxc or yxzc, we need to ensure
    # it has a single channel and we need to squeeze the channel dimension
    assert label.shape[-1] == 1, "Label image must have a single channel"
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


@validate_call
def measure_features(
    *,
    # Fractal managed parameters
    zarr_url: str,
    # Input parameters
    label_image_name: str,
    output_table_name: str = "region_props_features",
    features: list[SupportedFeatures] = Field(
        default_factory=lambda: [ShapeFeatures(), IntensityFeatures()]
    ),
    roi_tables: list[str] = Field(default_factory=list),
    andvanced_options: AdvancedOptions = AdvancedOptions(),
    overwrite: bool = False,
) -> None:
    """Extract region-properties features from an OME-Zarr image and save as a table.

    This task requires a label image to be present in the OME-Zarr container.

    Args:
        zarr_url (str): URL to the OME-Zarr container.
        label_image_name (str): Name of the label image to analyze.
        output_table_name (str): Name for the output feature table.
        features (list[SupportedFeatures]): List of feature configurations
            describing which properties to extract.
        roi_tables (list[str]): List of ROI table names to condition the
            feature extraction on. If empty, features will be extracted
            for the whole label image (2D) or volume (3D).
        advanced_options (AdvancedOptions): Advanced options for feature measurement.
        overwrite (bool): Whether to overwrite an existing feature table.
            Defaults to False.
    """
    logger.info(f"{zarr_url=}")

    ome_zarr = open_ome_zarr_container(zarr_url)

    if not overwrite and output_table_name in ome_zarr.list_tables():
        # This is already checked in ome_zarr.add_table, but we check it here
        # to fail early and avoid running the task unnecessarily
        raise FileExistsError(
            f"Table {output_table_name} already exists. "
            "Set overwrite=True to overwrite it."
        )

    image = ome_zarr.get_image(path=andvanced_options.level_path)
    regionprops_kwargs, channel_selection_models = _prepare_regionprops_kwargs(
        image=image,
        list_features=features,
        use_scaling=andvanced_options.use_scaling,
        use_cache=andvanced_options.use_cache,
    )

    iterator = setup_measurement_iterator(
        zarr_url=zarr_url,
        level_path=andvanced_options.level_path,
        label_image_name=label_image_name,
        channels=channel_selection_models,
        tables=roi_tables,
    )

    def extract_func(image: np.ndarray, label: np.ndarray, roi: Roi) -> dict:
        return region_props_features_func(image, label, roi, **regionprops_kwargs)

    feature_df = compute_measurement(func=extract_func, iterator=iterator)

    # Create a FeatureTable and add it to the OME-Zarr container
    feature_table = FeatureTable(
        table_data=feature_df, reference_label=label_image_name
    )
    ome_zarr.add_table(
        name=output_table_name,
        table=feature_table,
        overwrite=overwrite,
        backend=andvanced_options.table_backend,
    )
    logger.info(f"Feature table {output_table_name} added to OME-Zarr container.")
    return None


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=measure_features, logger_name=logger.name)

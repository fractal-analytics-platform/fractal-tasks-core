# Copyright 2022-2026 (C) BioVisionCenter, University of Zurich
"""Measure region-properties features from OME-Zarr label images."""

import logging
from typing import Annotated, Literal

import numpy as np
from fractal_tasks_utils.measurement import (
    compute_measurement,
    setup_measurement_iterator,
)
from ngio import Roi, open_ome_zarr_container
from ngio.tables import FeatureTable
from pydantic import BaseModel, Field, validate_call
from skimage import measure

logger = logging.getLogger("measure_features")


class ShapeFeatures(BaseModel):
    """Shape features extracted from regionprops."""

    type: Literal["ShapeFeatures"] = "ShapeFeatures"

    @property
    def property_names(self) -> list[str]:
        return [
            "area",
            "area_bbox",
            "axis_major_length",
            "axis_minor_length",
            "solidity",
        ]


class IntensityFeatures(BaseModel):
    """Intensity features extracted from regionprops."""

    type: Literal["IntensityFeatures"] = "IntensityFeatures"

    @property
    def property_names(self) -> list[str]:
        return [
            "mean_intensity",
            "max_intensity",
            "min_intensity",
            "std_intensity",
        ]


SupportedFeatures = Annotated[
    ShapeFeatures | IntensityFeatures,
    Field(discriminator="type"),
]


def region_props_features_func(
    image: np.ndarray,
    label: np.ndarray,
    roi: Roi,
    list_features: list[SupportedFeatures],
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

    properties = [
        "label",
    ]
    for feature in list_features:
        properties.extend(feature.property_names)

    # Perform region props extraction
    props = measure.regionprops_table(
        label,
        intensity_image=image,
        properties=properties,
    )
    # Add some more metadata columns, e.g. the ROI name
    num_regions = len(props["label"])
    props["region"] = [roi.get_name()] * num_regions
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

    iterator = setup_measurement_iterator(
        zarr_url=zarr_url,
        label_image_name=label_image_name,
    )

    def extract_func(image: np.ndarray, label: np.ndarray, roi: Roi) -> dict:
        return region_props_features_func(image, label, roi, list_features=features)

    feature_df = compute_measurement(func=extract_func, iterator=iterator)

    # Create a FeatureTable and add it to the OME-Zarr container
    feature_table = FeatureTable(
        table_data=feature_df, reference_label=label_image_name
    )
    ome_zarr.add_table(name=output_table_name, table=feature_table, overwrite=overwrite)
    logger.info(f"Feature table {output_table_name} added to OME-Zarr container.")
    return None


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=measure_features, logger_name=logger.name)

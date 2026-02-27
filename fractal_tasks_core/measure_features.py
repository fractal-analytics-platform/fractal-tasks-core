# Copyright 2022-2026 (C) BioVisionCenter, University of Zurich
"""Measure region-properties features from OME-Zarr label images."""

import logging

import numpy as np
import pandas as pd
from ngio import Roi, open_ome_zarr_container
from ngio.experimental.iterators import FeatureExtractorIterator
from ngio.tables import FeatureTable
from ngio.transforms import ZoomTransform
from pydantic import BaseModel, validate_call
from skimage import measure

logger = logging.getLogger("measure_features")


class ShapeFeatures(BaseModel):
    """Shape features extracted from regionprops."""

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

    @property
    def property_names(self) -> list[str]:
        return [
            "mean_intensity",
            "max_intensity",
            "min_intensity",
            "std_intensity",
        ]


def region_props_features_func(
    image: np.ndarray,
    label: np.ndarray,
    roi: Roi,
    list_features: list[ShapeFeatures | IntensityFeatures],
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


def join_tables(
    tables: list[dict[str, list]], index_key: str = "label"
) -> pd.DataFrame:
    """Join a list of tables (dictionaries) into a single DataFrame."""
    assert len(tables) >= 1, "At least one table is required"
    out_dict = {}
    for table in tables:
        for key, value in table.items():
            if key not in out_dict:
                out_dict[key] = []
            out_dict[key].extend(value)

    df = pd.DataFrame(out_dict)
    df = df.set_index(index_key)
    return df


@validate_call
def measure_features(
    *,
    # Fractal managed parameters
    zarr_url: str,
    # Input parameters
    label_image_name: str,
    output_table_name: str = "region_props_features",
    features: list[ShapeFeatures | IntensityFeatures],
    overwrite: bool = False,
) -> None:
    """Extract region-properties features from an OME-Zarr image and save as a table.

    This task requires a label image to be present in the OME-Zarr container.

    Args:
        zarr_url (str): URL to the OME-Zarr container.
        label_image_name (str): Name of the label image to analyze.
        output_table_name (str): Name for the output feature table.
        features (list[ShapeFeatures | IntensityFeatures]): List of feature
            configurations describing which properties to extract.
        overwrite (bool): Whether to overwrite an existing feature table.
            Defaults to False.
    """
    logger.info(f"{zarr_url=}")

    # Open the OME-Zarr container
    ome_zarr = open_ome_zarr_container(zarr_url)
    logger.info(f"{ome_zarr=}")

    # Get the image at the highest resolution available
    image = ome_zarr.get_image()
    logger.info(f"{image=}")

    # Get the label image at the closest resolution to the image.
    # If the label image doesn't have an exact match in pixel size, we
    # get the closest one (strict=False)
    label_image = ome_zarr.get_label(
        name=label_image_name, pixel_size=image.pixel_size, strict=False
    )
    logger.info(f"{label_image=}")

    if not overwrite and output_table_name in ome_zarr.list_tables():
        # This is already checked in ome_zarr.add_table, but we check it here
        # to fail early and avoid running the task unnecessarily
        raise FileExistsError(
            f"Table {output_table_name} already exists. "
            "Set overwrite=True to overwrite it."
        )

    # Some of the features in regionprops fail if the image has a singleton z.
    # By setting the axes_order to yxc we ensure to squeeze the z dimension
    # for 2D images. For 3D images, we keep the z dimension.
    axes_order = "yxc" if image.is_2d else "yxzc"

    # Since the label image might be at a different resolution than the image,
    # we need to rescale it to match the image resolution
    label_zoom_transform = ZoomTransform(
        input_image=label_image,
        target_image=image,
        order="nearest",
    )

    # Create an iterator to process the image and extract features
    iterator = FeatureExtractorIterator(
        input_image=image,
        input_label=label_image,
        axes_order=axes_order,
        label_transforms=[label_zoom_transform],
    )
    # We can iterate by_zyx even in a 2D case by setting strict to False.
    # This way, the iterator will return 3D images if the OME-Zarr contains 3D data,
    # but return 2D images otherwise
    iterator = iterator.by_zyx(strict=False)

    # Core processing loop
    tables = []
    for input_data, label_data, roi in iterator.iter_as_numpy():
        _table_dict = region_props_features_func(
            image=input_data,
            label=label_data,
            roi=roi,
            list_features=features,
        )
        # FeatureExtractorIterator does not handle writing, so we collect
        # the tables and write them at the end
        tables.append(_table_dict)

    # Convert the tables to a DataFrame
    feature_df = join_tables(tables, index_key="label")

    # Create a FeatureTable and add it to the OME-Zarr container
    feature_table = FeatureTable(
        table_data=feature_df, reference_label=label_image_name
    )
    # Save the DataFrame as a table in the OME-Zarr container
    ome_zarr.add_table(name=output_table_name, table=feature_table, overwrite=overwrite)
    logger.info(f"Feature table {output_table_name} added to OME-Zarr container.")
    return None


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=measure_features, logger_name=logger.name)

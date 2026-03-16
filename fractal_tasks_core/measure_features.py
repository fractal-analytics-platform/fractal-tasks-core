# Copyright 2022-2026 (C) BioVisionCenter, University of Zurich
"""Measure region-properties features from OME-Zarr label images."""

import logging

import numpy as np
from fractal_tasks_utils.measurement import (
    compute_measurement,
    setup_measurement_iterator,
)
from ngio import Roi, open_ome_zarr_container
from ngio.tables import FeatureTable
from pydantic import Field, validate_call

from fractal_tasks_core._measure_features_utils import (
    AdvancedOptions,
    IntensityFeatures,
    ShapeFeatures,
    SupportedFeatures,
    _prepare_regionprops_kwargs,
    region_props_features_func,
)

logger = logging.getLogger("measure_features")


@validate_call
def measure_features(
    *,
    # Fractal managed parameters
    zarr_url: str,
    # Input parameters
    input_label_name: str,
    output_table_name: str = "region_props_features",
    features: list[SupportedFeatures] = Field(
        default_factory=lambda: [ShapeFeatures(), IntensityFeatures()]
    ),
    roi_tables: list[str] = Field(default_factory=list),
    advanced_options: AdvancedOptions = AdvancedOptions(),
    overwrite: bool = False,
) -> None:
    """Extract region-properties features from an OME-Zarr image and save as a table.

    This task requires a label image to be present in the OME-Zarr container.

    Args:
        zarr_url: URL to the OME-Zarr container.
        input_label_name: Name of the segmentation label image to measure
            (must already exist in the OME-Zarr).
        output_table_name: Name for the output feature table.
        features: List of feature configurations describing which properties
            to extract.
        roi_tables: List of ROI table names to condition the feature extraction
            on. If empty, features will be extracted for the whole label image
            (2D) or volume (3D).
        advanced_options: Advanced options for feature measurement.
        overwrite: Whether to overwrite an existing feature table.
            Defaults to False.
    """
    logger.info(f"{zarr_url=}")

    ome_zarr = open_ome_zarr_container(zarr_url)

    _seen_types: set[str] = set()
    for _feat in features:
        _feat_type = type(_feat).__name__
        if _feat_type in _seen_types:
            raise ValueError(
                f"Duplicate feature type '{_feat_type}' in features list. "
                "Each feature type may only appear once."
            )
        _seen_types.add(_feat_type)

    if not overwrite and output_table_name in ome_zarr.list_tables():
        # This is already checked in ome_zarr.add_table, but we check it here
        # to fail early and avoid running the task unnecessarily
        raise FileExistsError(
            f"Table {output_table_name} already exists. "
            "Set overwrite=True to overwrite it."
        )

    image = ome_zarr.get_image(path=advanced_options.level_path)
    regionprops_kwargs, channel_selection_models = _prepare_regionprops_kwargs(
        image=image,
        list_features=features,
        use_scaling=advanced_options.use_scaling,
        use_cache=advanced_options.use_cache,
    )

    iterator = setup_measurement_iterator(
        zarr_url=zarr_url,
        level_path=advanced_options.level_path,
        label_image_name=input_label_name,
        channels=channel_selection_models,
        roi_table_names=roi_tables if roi_tables else None,
    )

    def extract_func(image: np.ndarray, label: np.ndarray, roi: Roi) -> dict:
        return region_props_features_func(image, label, roi, **regionprops_kwargs)

    feature_df = compute_measurement(measurement_func=extract_func, iterator=iterator)

    # Create a FeatureTable and add it to the OME-Zarr container
    feature_table = FeatureTable(
        table_data=feature_df, reference_label=input_label_name
    )
    ome_zarr.add_table(
        name=output_table_name,
        table=feature_table,
        overwrite=overwrite,
        backend=advanced_options.table_backend,
    )
    logger.info(f"Feature table {output_table_name} added to OME-Zarr container.")
    return None


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=measure_features, logger_name=logger.name)

# Copyright 2022-2026 (C) BioVisionCenter, University of Zurich
"""Pydantic models for the import_ome_zarr task."""

from pydantic import BaseModel

from fractal_tasks_core._utils import AVAILABLE_TABLE_BACKENDS


class GridRoiTable(BaseModel):
    """Configuration for building a `grid_ROI_table`."""

    grid_y_shape: int = 2
    """
    Number of ROIs along the Y axis. The image is split into a `grid_y_shape` by
    `grid_x_shape` grid of ROIs (e.g. the default 2 by 2 produces 4 ROIs).
    """

    grid_x_shape: int = 2
    """
    Number of ROIs along the X axis. The image is split into a `grid_y_shape` by
    `grid_x_shape` grid of ROIs (e.g. the default 2 by 2 produces 4 ROIs).
    """


class AdvancedOptions(BaseModel):
    """Advanced options for importing an OME-Zarr."""

    update_omero_metadata: bool = False
    """
    Whether to update Omero-channels metadata with channel labels and
    wavelength ids to make them compatible with some downstream Fractal tasks.
    """

    add_image_roi_table: bool = False
    """
    Whether to add an `image_ROI_table` table to each image, with a single ROI
    covering the whole image.
    """

    grid_roi_table: GridRoiTable | None = None
    """
    If set, add a `grid_ROI_table` table to each image, with the image split
    into a rectangular grid of ROIs of the given shape. If `None`, no
    `grid_ROI_table` is created.
    """

    table_backend: AVAILABLE_TABLE_BACKENDS = "csv"
    """
    Table backend to use for the new ROI tables. Defaults to "csv".
    """

    overwrite: bool = False
    """
    Whether new ROI tables (added when `add_image_roi_table=True` and/or
    `grid_roi_table` is not `None`) can overwrite existing ones.
    """

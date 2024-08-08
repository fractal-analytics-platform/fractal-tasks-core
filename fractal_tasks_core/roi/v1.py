# Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Tommaso Comparin <tommaso.comparin@exact-lab.it>
# Joel Lüthi <joel.luethi@uzh.ch>
#
# This file is part of Fractal and was originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
"""
Functions to produce/process ROI tables.
"""
import logging
import math
from typing import Sequence

import anndata as ad
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def prepare_FOV_ROI_table(
    df: pd.DataFrame, metadata: tuple[str, ...] = ("time",)
) -> ad.AnnData:
    """
    Prepare an AnnData table for fields-of-view ROIs.

    Args:
        df:
            Input dataframe, possibly prepared through
            `parse_yokogawa_metadata`.
        metadata:
            Columns of `df` to be stored (if present) into AnnData table `obs`.
    """

    # Make a local copy of the dataframe, to avoid SettingWithCopyWarning
    df = df.copy()

    # Convert DataFrame index to str, to avoid
    # >> ImplicitModificationWarning: Transforming to str index
    # when creating AnnData object.
    # Do this in the beginning to allow concatenation with e.g. time
    df.index = df.index.astype(str)

    # Obtain box size in physical units
    df = df.assign(len_x_micrometer=df.x_pixel * df.pixel_size_x)
    df = df.assign(len_y_micrometer=df.y_pixel * df.pixel_size_y)
    df = df.assign(len_z_micrometer=df.z_pixel * df.pixel_size_z)

    # Select only the numeric positional columns needed to define ROIs
    # (to avoid) casting things like the data column to float32
    # or to use unnecessary columns like bit_depth
    positional_columns = [
        "x_micrometer",
        "y_micrometer",
        "z_micrometer",
        "len_x_micrometer",
        "len_y_micrometer",
        "len_z_micrometer",
        "x_micrometer_original",
        "y_micrometer_original",
    ]

    # Assign dtype explicitly, to avoid
    # >> UserWarning: X converted to numpy array with dtype float64
    # when creating AnnData object
    df_roi = df.loc[:, positional_columns].astype(np.float32)

    # Create an AnnData object directly from the DataFrame
    adata = ad.AnnData(X=df_roi)

    # Reset origin of the FOV ROI table, so that it matches with the well
    # origin
    adata = reset_origin(adata)

    # Save any metadata that is specified to the obs df
    for col in metadata:
        if col in df:
            # Cast all metadata to str.
            # Reason: AnnData Zarr writers don't support all pandas types.
            # e.g. pandas.core.arrays.datetimes.DatetimeArray can't be written
            adata.obs[col] = df[col].astype(str)

    # Rename rows and columns: Maintain FOV indices from the dataframe
    # (they are already enforced to be unique by Pandas and may contain
    # information for the user, as they are based on the filenames)
    adata.obs_names = "FOV_" + adata.obs.index
    adata.var_names = list(map(str, df_roi.columns))

    return adata


def prepare_well_ROI_table(
    df: pd.DataFrame, metadata: tuple[str, ...] = ("time",)
) -> ad.AnnData:
    """
    Prepare an AnnData table with a single well ROI.

    Args:
        df:
            Input dataframe, possibly prepared through
            `parse_yokogawa_metadata`.
        metadata:
            Columns of `df` to be stored (if present) into AnnData table `obs`.
    """

    # Make a local copy of the dataframe, to avoid SettingWithCopyWarning
    df = df.copy()

    # Convert DataFrame index to str, to avoid
    # >> ImplicitModificationWarning: Transforming to str index
    # when creating AnnData object.
    # Do this in the beginning to allow concatenation with e.g. time
    df.index = df.index.astype(str)

    # Calculate bounding box extents in physical units
    for mu in ["x", "y", "z"]:
        # Obtain per-FOV properties in physical units.
        # NOTE: a FOV ROI is defined here as the interval [min_micrometer,
        # max_micrometer], with max_micrometer=min_micrometer+len_micrometer
        min_micrometer = df[f"{mu}_micrometer"]
        len_micrometer = df[f"{mu}_pixel"] * df[f"pixel_size_{mu}"]
        max_micrometer = min_micrometer + len_micrometer
        # Obtain well bounding box, in physical units
        min_min_micrometer = min_micrometer.min()
        max_max_micrometer = max_micrometer.max()
        df[f"{mu}_micrometer"] = min_min_micrometer
        df[f"len_{mu}_micrometer"] = max_max_micrometer - min_min_micrometer

    # Select only the numeric positional columns needed to define ROIs
    # (to avoid) casting things like the data column to float32
    # or to use unnecessary columns like bit_depth
    positional_columns = [
        "x_micrometer",
        "y_micrometer",
        "z_micrometer",
        "len_x_micrometer",
        "len_y_micrometer",
        "len_z_micrometer",
    ]

    # Assign dtype explicitly, to avoid
    # >> UserWarning: X converted to numpy array with dtype float64
    # when creating AnnData object
    df_roi = df.iloc[0:1, :].loc[:, positional_columns].astype(np.float32)

    # Create an AnnData object directly from the DataFrame
    adata = ad.AnnData(X=df_roi)

    # Reset origin of the single-entry well ROI table
    adata = reset_origin(adata)

    # Save any metadata that is specified to the obs df
    for col in metadata:
        if col in df:
            # Cast all metadata to str.
            # Reason: AnnData Zarr writers don't support all pandas types.
            # e.g. pandas.core.arrays.datetimes.DatetimeArray can't be written
            adata.obs[col] = df[col].astype(str)

    # Rename rows and columns: Maintain FOV indices from the dataframe
    # (they are already enforced to be unique by Pandas and may contain
    # information for the user, as they are based on the filenames)
    adata.obs_names = "well_" + adata.obs.index
    adata.var_names = list(map(str, df_roi.columns))

    return adata


def convert_ROIs_from_3D_to_2D(
    adata: ad.AnnData,
    pixel_size_z: float,
) -> ad.AnnData:
    """
    TBD

    Note that this function is only relevant when the ROIs in adata span the
    whole extent of the Z axis.
    TODO: check this explicitly.

    Args:
        adata: TBD
        pixel_size_z: TBD
    """

    # Compress a 3D stack of images to a single Z plane,
    # with thickness equal to pixel_size_z
    df = adata.to_df()
    df["len_z_micrometer"] = pixel_size_z

    # Assign dtype explicitly, to avoid
    # >> UserWarning: X converted to numpy array with dtype float64
    # when creating AnnData object
    df = df.astype(np.float32)

    # Create an AnnData object directly from the DataFrame
    new_adata = ad.AnnData(X=df)

    # Rename rows and columns
    new_adata.obs_names = adata.obs_names
    new_adata.var_names = list(map(str, df.columns))

    return new_adata


def convert_ROI_table_to_indices(
    ROI: ad.AnnData,
    full_res_pxl_sizes_zyx: Sequence[float],
    level: int = 0,
    coarsening_xy: int = 2,
    cols_xyz_pos: Sequence[str] = [
        "x_micrometer",
        "y_micrometer",
        "z_micrometer",
    ],
    cols_xyz_len: Sequence[str] = [
        "len_x_micrometer",
        "len_y_micrometer",
        "len_z_micrometer",
    ],
) -> list[list[int]]:
    """
    Convert a ROI AnnData table into integer array indices.

    Args:
        ROI: AnnData table with list of ROIs.
        full_res_pxl_sizes_zyx:
            Physical-unit pixel ZYX sizes at the full-resolution pyramid level.
        level: Pyramid level.
        coarsening_xy: Linear coarsening factor in the YX plane.
        cols_xyz_pos: Column names for XYZ ROI positions.
        cols_xyz_len: Column names for XYZ ROI edges.

    Raises:
        ValueError:
            If any of the array indices is negative.

    Returns:
        Nested list of indices. The main list has one item per ROI. Each ROI
            item is a list of six integers as in `[start_z, end_z, start_y,
            end_y, start_x, end_x]`. The array-index interval for a given ROI
            is `start_x:end_x` along X, and so on for Y and Z.
    """
    # Handle empty ROI table
    if len(ROI) == 0:
        return []

    # Set pyramid-level pixel sizes
    pxl_size_z, pxl_size_y, pxl_size_x = full_res_pxl_sizes_zyx
    prefactor = coarsening_xy**level
    pxl_size_x *= prefactor
    pxl_size_y *= prefactor

    x_pos, y_pos, z_pos = cols_xyz_pos[:]
    x_len, y_len, z_len = cols_xyz_len[:]

    list_indices = []
    for ROI_name in ROI.obs_names:
        # Extract data from anndata table
        x_micrometer = ROI[ROI_name, x_pos].X[0, 0]
        y_micrometer = ROI[ROI_name, y_pos].X[0, 0]
        z_micrometer = ROI[ROI_name, z_pos].X[0, 0]
        len_x_micrometer = ROI[ROI_name, x_len].X[0, 0]
        len_y_micrometer = ROI[ROI_name, y_len].X[0, 0]
        len_z_micrometer = ROI[ROI_name, z_len].X[0, 0]

        # Identify indices along the three dimensions
        start_x = x_micrometer / pxl_size_x
        end_x = (x_micrometer + len_x_micrometer) / pxl_size_x
        start_y = y_micrometer / pxl_size_y
        end_y = (y_micrometer + len_y_micrometer) / pxl_size_y
        start_z = z_micrometer / pxl_size_z
        end_z = (z_micrometer + len_z_micrometer) / pxl_size_z
        indices = [start_z, end_z, start_y, end_y, start_x, end_x]

        # Round indices to lower integer
        indices = list(map(round, indices))

        # Fail for negative indices
        if min(indices) < 0:
            raise ValueError(
                f"ROI {ROI_name} converted into negative array indices.\n"
                f"ZYX position: {z_micrometer}, {y_micrometer}, "
                f"{x_micrometer}\n"
                f"ZYX pixel sizes: {pxl_size_z}, {pxl_size_y}, "
                f"{pxl_size_x} ({level=})\n"
                "Hint: As of fractal-tasks-core v0.12, FOV/well ROI "
                "tables with non-zero origins (e.g. the ones created with "
                "v0.11) are not supported."
            )

        # Append ROI indices to to list
        list_indices.append(indices[:])

    return list_indices


def empty_bounding_box_table() -> pd.DataFrame:
    """
    Construct an empty bounding-box ROI table of given shape.

    This function mirrors the functionality of `array_to_bounding_box_table`,
    for the specific case where the array includes no label. The advantages of
    this function are that:

    1. It does not require computing a whole array of zeros;
    2. We avoid hardcoding column names in the task functions.

    Returns:
        DataFrame with no rows, and with columns corresponding to the output of
            `array_to_bounding_box_table`.
    """

    df_columns = [
        "x_micrometer",
        "y_micrometer",
        "z_micrometer",
        "len_x_micrometer",
        "len_y_micrometer",
        "len_z_micrometer",
    ]
    df = pd.DataFrame(columns=[x for x in df_columns] + ["label"])
    return df


def array_to_bounding_box_table(
    mask_array: np.ndarray,
    pxl_sizes_zyx: list[float],
    origin_zyx: tuple[int, int, int] = (0, 0, 0),
) -> pd.DataFrame:
    """
    Construct bounding-box ROI table for a mask array.

    Args:
        mask_array: Original array to construct bounding boxes.
        pxl_sizes_zyx: Physical-unit pixel ZYX sizes.
        origin_zyx: Shift ROI origin by this amount of ZYX pixels.

    Returns:
        DataFrame with each line representing the bounding-box ROI that
            corresponds to a unique value of `mask_array`. ROI properties are
            expressed in physical units (with columns defined as elsewhere this
            module - see e.g. `prepare_well_ROI_table`), and positions are
            optionally shifted (if `origin_zyx` is set). An additional column
            `label` keeps track of the `mask_array` value corresponding to each
            ROI.
    """

    pxl_sizes_zyx_array = np.array(pxl_sizes_zyx)
    z_origin, y_origin, x_origin = origin_zyx[:]

    labels = np.unique(mask_array)
    labels = labels[labels > 0]
    elem_list = []
    for label in labels:
        # Compute bounding box
        label_match = np.where(mask_array == label)
        zmin, ymin, xmin = np.min(label_match, axis=1) * pxl_sizes_zyx_array
        zmax, ymax, xmax = (
            np.max(label_match, axis=1) + 1
        ) * pxl_sizes_zyx_array

        # Compute bounding-box edges
        length_x = xmax - xmin
        length_y = ymax - ymin
        length_z = zmax - zmin

        # Shift origin
        zmin += z_origin * pxl_sizes_zyx[0]
        ymin += y_origin * pxl_sizes_zyx[1]
        xmin += x_origin * pxl_sizes_zyx[2]

        elem_list.append((xmin, ymin, zmin, length_x, length_y, length_z))

    df_columns = [
        "x_micrometer",
        "y_micrometer",
        "z_micrometer",
        "len_x_micrometer",
        "len_y_micrometer",
        "len_z_micrometer",
    ]

    if len(elem_list) == 0:
        df = pd.DataFrame(columns=[x for x in df_columns] + ["label"])
    else:
        df = pd.DataFrame(np.array(elem_list), columns=df_columns)
        df["label"] = labels

    return df


def convert_indices_to_regions(
    index: list[int],
) -> tuple[slice, slice, slice]:
    """
    Converts index tuples to region tuple

    Args:
        index: Tuple containing 6 entries of (z_start, z_end, y_start,
            y_end, x_start, x_end).

    Returns:
        region: tuple of three slices (ZYX)
    """
    return (
        slice(index[0], index[1]),
        slice(index[2], index[3]),
        slice(index[4], index[5]),
    )


def reset_origin(
    ROI_table: ad.AnnData,
    x_pos: str = "x_micrometer",
    y_pos: str = "y_micrometer",
    z_pos: str = "z_micrometer",
) -> ad.AnnData:
    """
    Return a copy of a ROI table, with shifted-to-zero origin for some columns.

    Args:
        ROI_table: Original ROI table.
        x_pos: Name of the column with X position of ROIs.
        y_pos: Name of the column with Y position of ROIs.
        z_pos: Name of the column with Z position of ROIs.

    Returns:
        A copy of the `ROI_table` AnnData table, where values of `x_pos`,
            `y_pos` and `z_pos` columns have been shifted by their minimum
            values.
    """
    new_table = ROI_table.copy()

    origin_x = min(new_table[:, x_pos].X[:, 0])
    origin_y = min(new_table[:, y_pos].X[:, 0])
    origin_z = min(new_table[:, z_pos].X[:, 0])

    for FOV in new_table.obs_names:
        new_table[FOV, x_pos] = new_table[FOV, x_pos].X[0, 0] - origin_x
        new_table[FOV, y_pos] = new_table[FOV, y_pos].X[0, 0] - origin_y
        new_table[FOV, z_pos] = new_table[FOV, z_pos].X[0, 0] - origin_z

    return new_table


def is_standard_roi_table(table: str) -> bool:
    """
    True if the name of the table contains one of the standard Fractal tables

    If a table name is well_ROI_table, FOV_ROI_table or contains either of the
    two (e.g. registered_FOV_ROI_table), this function returns True.

    Args:
        table: table name

    Returns:
        bool of whether it's a standard ROI table

    """
    if "well_ROI_table" in table:
        return True
    elif "FOV_ROI_table" in table:
        return True
    else:
        return False


def get_single_image_ROI(
    array_shape: tuple[int, int, int],
    pixels_ZYX: list[float],
) -> ad.AnnData:
    """
    Produce a table with a single ROI that covers the whole array

    TODO: make this flexible with respect to the presence/absence of Z.

    Args:
        array_shape: ZYX shape of the image array.
        pixels_ZYX: ZYX pixel sizes in micrometers.

    Returns:
        An `AnnData` table with a single ROI.
    """
    shape_z, shape_y, shape_x = array_shape[-3:]
    ROI_table = ad.AnnData(
        X=np.array(
            [
                [
                    0.0,
                    0.0,
                    0.0,
                    shape_x * pixels_ZYX[2],
                    shape_y * pixels_ZYX[1],
                    shape_z * pixels_ZYX[0],
                ],
            ],
            dtype=np.float32,
        )
    )
    ROI_table.obs_names = ["image_1"]
    ROI_table.var_names = [
        "x_micrometer",
        "y_micrometer",
        "z_micrometer",
        "len_x_micrometer",
        "len_y_micrometer",
        "len_z_micrometer",
    ]
    return ROI_table


def get_image_grid_ROIs(
    array_shape: tuple[int, int, int],
    pixels_ZYX: list[float],
    grid_YX_shape: tuple[int, int],
) -> ad.AnnData:
    """
    Produce a table with ROIS placed on a rectangular grid.

    The main goal of this ROI grid is to allow processing of smaller subset of
    the whole array.

    In a specific case (that is, if the image array was obtained by stitching
    together a set of FOVs placed on a regular grid), the ROIs correspond to
    the original FOVs.

    TODO: make this flexible with respect to the presence/absence of Z.

    Args:
        array_shape: ZYX shape of the image array.
        pixels_ZYX: ZYX pixel sizes in micrometers.
        grid_YX_shape:

    Returns:
        An `AnnData` table with a single ROI.
    """
    shape_z, shape_y, shape_x = array_shape[-3:]
    grid_size_y, grid_size_x = grid_YX_shape[:]
    X = []
    obs_names = []
    counter = 0
    start_z = 0
    len_z = shape_z

    # Find minimal len_y that covers [0,shape_y] with grid_size_y intervals
    len_y = math.ceil(shape_y / grid_size_y)
    len_x = math.ceil(shape_x / grid_size_x)
    for ind_y in range(grid_size_y):
        start_y = ind_y * len_y
        tmp_len_y = min(shape_y, start_y + len_y) - start_y
        for ind_x in range(grid_size_x):
            start_x = ind_x * len_x
            tmp_len_x = min(shape_x, start_x + len_x) - start_x
            X.append(
                [
                    start_x * pixels_ZYX[2],
                    start_y * pixels_ZYX[1],
                    start_z * pixels_ZYX[0],
                    tmp_len_x * pixels_ZYX[2],
                    tmp_len_y * pixels_ZYX[1],
                    len_z * pixels_ZYX[0],
                ]
            )
            counter += 1
            obs_names.append(f"ROI_{counter}")
    ROI_table = ad.AnnData(X=np.array(X, dtype=np.float32))
    ROI_table.obs_names = obs_names
    ROI_table.var_names = [
        "x_micrometer",
        "y_micrometer",
        "z_micrometer",
        "len_x_micrometer",
        "len_y_micrometer",
        "len_z_micrometer",
    ]
    return ROI_table


def create_roi_table_from_df_list(
    bbox_dataframe_list: list[pd.DataFrame],
) -> ad.AnnData:
    """
    Creates an AnnData ROI table from a list of bounding-box tables

    This function handles empty bbox lists, ensures that it has unique entries
    per label (and to address #810, it handles duplicate labels by only
    keeping the first entry for each label) & converts it to an AnnData table
    with a label column in obs.

    Args:
        bbox_dataframe_list: List of bounding box dataframes. All dataframes
            are expected to have the same columns and they usually are:
            x_micrometer, y_micrometer, z_micrometer, len_x_micrometer,
            len_y_micrometer, len_z_micrometer, label. The label column is
            required.

    Returns:
        An `AnnData` table with all the ROIs.
    """
    # Handle the case where `bbox_dataframe_list` is empty (typically
    # because list_indices is also empty)
    if len(bbox_dataframe_list) == 0:
        bbox_dataframe_list = [empty_bounding_box_table()]
    # Concatenate all ROI dataframes
    df_well = pd.concat(bbox_dataframe_list, axis=0, ignore_index=True)

    # Drop duplicates based on the 'label' column, keeping only the first
    # occurrence (see #810 for details)
    df_well = df_well.drop_duplicates(subset="label", keep="first")

    # Extract labels and drop them from df_well
    labels = pd.DataFrame(df_well["label"].astype(str)).reset_index(drop=True)

    # Check that there are only unique labels. Should be ensured by check above
    if len(labels["label"]) != len(labels["label"].unique()):
        raise ValueError(
            "The output ROI table contains duplicate entries for labels: "
            f"It contains {len(labels['label'])} entries, but only "
            f"{len(labels['label'].unique())} unique labels"
        )

    df_well.index = labels["label"]
    df_well.drop(labels=["label"], axis=1, inplace=True)
    # Convert all to float (warning: some would be int, in principle)
    bbox_dtype = np.float32
    df_well = df_well.astype(bbox_dtype)
    # Convert to anndata
    bbox_table = ad.AnnData(df_well)
    bbox_table.obs = labels
    return bbox_table

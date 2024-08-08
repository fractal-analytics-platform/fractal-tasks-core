# Copyright 2024 (C) BioVisionCenter
#
# Original authors:
# Joel Lüthi <joel.luethi@uzh.ch>
#
# This file is part of Fractal
"""Utils functions for registration"""
import copy

import anndata as ad
import dask.array as da
import numpy as np
import pandas as pd
from image_registration import chi2_shift


def calculate_physical_shifts(
    shifts: np.array,
    level: int,
    coarsening_xy: int,
    full_res_pxl_sizes_zyx: list[float],
) -> list[float]:
    """
    Calculates shifts in physical units based on pixel shifts

    Args:
        shifts: array of shifts, zyx or yx
        level: resolution level
        coarsening_xy: coarsening factor between levels
        full_res_pxl_sizes_zyx: pixel sizes in physical units as zyx

    Returns:
        shifts in physical units as zyx
    """

    curr_pixel_size = np.array(full_res_pxl_sizes_zyx) * coarsening_xy**level
    if len(shifts) == 3:
        shifts_physical = shifts * curr_pixel_size
    elif len(shifts) == 2:
        shifts_physical = [
            0,
            shifts[0] * curr_pixel_size[1],
            shifts[1] * curr_pixel_size[2],
        ]
    else:
        raise ValueError(
            f"Wrong input for calculate_physical_shifts ({shifts=})"
        )
    return shifts_physical


def get_ROI_table_with_translation(
    ROI_table: ad.AnnData,
    new_shifts: dict[str, list[float]],
) -> ad.AnnData:
    """
    Adds translation columns to a ROI table

    Args:
        ROI_table: Fractal ROI table
        new_shifts: zyx list of shifts

    Returns:
        Fractal ROI table with 3 additional columns for calculated translations
    """

    shift_table = pd.DataFrame(new_shifts).T
    shift_table.columns = ["translation_z", "translation_y", "translation_x"]
    shift_table = shift_table.rename_axis("FieldIndex")
    new_roi_table = ROI_table.to_df().merge(
        shift_table, left_index=True, right_index=True
    )
    if len(new_roi_table) != len(ROI_table):
        raise ValueError(
            "New ROI table with registration info has a "
            f"different length ({len(new_roi_table)=}) "
            f"from the original ROI table ({len(ROI_table)=})"
        )

    adata = ad.AnnData(X=new_roi_table.astype(np.float32))
    adata.obs_names = new_roi_table.index
    adata.var_names = list(map(str, new_roi_table.columns))
    return adata


# Helper functions
def add_zero_translation_columns(ad_table: ad.AnnData):
    """
    Add three zero-filled columns (`translation_{x,y,z}`) to an AnnData table.
    """
    columns = ["translation_z", "translation_y", "translation_x"]
    if ad_table.var.index.isin(columns).any().any():
        raise ValueError(
            "The roi table already contains translation columns. Did you "
            "enter a wrong reference acquisition?"
        )
    df = pd.DataFrame(np.zeros([len(ad_table), 3]), columns=columns)
    df.index = ad_table.obs.index
    ad_new = ad.concat([ad_table, ad.AnnData(df)], axis=1)
    return ad_new


def calculate_min_max_across_dfs(tables_list):
    # Initialize dataframes to store the max and min values
    max_df = pd.DataFrame(
        index=tables_list[0].index, columns=tables_list[0].columns
    )
    min_df = pd.DataFrame(
        index=tables_list[0].index, columns=tables_list[0].columns
    )

    # Loop through the tables and calculate max and min values
    for table in tables_list:
        max_df = pd.DataFrame(
            np.maximum(max_df.values, table.values),
            columns=max_df.columns,
            index=max_df.index,
        )
        min_df = pd.DataFrame(
            np.minimum(min_df.values, table.values),
            columns=min_df.columns,
            index=min_df.index,
        )

    return max_df, min_df


def apply_registration_to_single_ROI_table(
    roi_table: ad.AnnData,
    max_df: pd.DataFrame,
    min_df: pd.DataFrame,
) -> ad.AnnData:
    """
    Applies the registration to a ROI table

    Calculates the new position as: p = position + max(shift, 0) - own_shift
    Calculates the new len as: l = len - max(shift, 0) + min(shift, 0)

    Args:
        roi_table: AnnData table which contains a Fractal ROI table.
            Rows are ROIs
        max_df: Max translation shift in z, y, x for each ROI. Rows are ROIs,
            columns are translation_z, translation_y, translation_x
        min_df: Min translation shift in z, y, x for each ROI. Rows are ROIs,
            columns are translation_z, translation_y, translation_x
    Returns:
        ROI table where all ROIs are registered to the smallest common area
        across all acquisitions.
    """
    roi_table = copy.deepcopy(roi_table)
    rois = roi_table.obs.index
    if (rois != max_df.index).all() or (rois != min_df.index).all():
        raise ValueError(
            "ROI table and max & min translation need to contain the same "
            f"ROIS, but they were {rois=}, {max_df.index=}, {min_df.index=}"
        )

    for roi in rois:
        roi_table[[roi], ["z_micrometer"]] = (
            roi_table[[roi], ["z_micrometer"]].X
            + float(max_df.loc[roi, "translation_z"])
            - roi_table[[roi], ["translation_z"]].X
        )
        roi_table[[roi], ["y_micrometer"]] = (
            roi_table[[roi], ["y_micrometer"]].X
            + float(max_df.loc[roi, "translation_y"])
            - roi_table[[roi], ["translation_y"]].X
        )
        roi_table[[roi], ["x_micrometer"]] = (
            roi_table[[roi], ["x_micrometer"]].X
            + float(max_df.loc[roi, "translation_x"])
            - roi_table[[roi], ["translation_x"]].X
        )
        # This calculation only works if all ROIs are the same size initially!
        roi_table[[roi], ["len_z_micrometer"]] = (
            roi_table[[roi], ["len_z_micrometer"]].X
            - float(max_df.loc[roi, "translation_z"])
            + float(min_df.loc[roi, "translation_z"])
        )
        roi_table[[roi], ["len_y_micrometer"]] = (
            roi_table[[roi], ["len_y_micrometer"]].X
            - float(max_df.loc[roi, "translation_y"])
            + float(min_df.loc[roi, "translation_y"])
        )
        roi_table[[roi], ["len_x_micrometer"]] = (
            roi_table[[roi], ["len_x_micrometer"]].X
            - float(max_df.loc[roi, "translation_x"])
            + float(min_df.loc[roi, "translation_x"])
        )
    return roi_table


def chi2_shift_out(img_ref, img_cycle_x) -> list[np.ndarray]:
    """
    Helper function to get the output of chi2_shift into the same format as
    phase_cross_correlation. Calculates the shift between two images using
    the chi2_shift method.

    Args:
        img_ref (np.ndarray): First image.
        img_cycle_x (np.ndarray): Second image.

    Returns:
        List containing numpy array of shift in y and x direction.
    """
    x, y, a, b = chi2_shift(np.squeeze(img_ref), np.squeeze(img_cycle_x))

    """
    Running into issues when using direct float output for fractal.
    When rounding to integer and using integer dtype, it typically works
    but for some reasons fails when run over a whole 384 well plate (but
    the well where it fails works fine when run alone). For now, rounding
    to integer, but still using float64 dtype (like the scikit-image
    phase cross correlation function) seems to be the safest option.
    """
    shifts = np.array([-np.round(y), -np.round(x)], dtype="float64")
    # return as a list to adhere to the phase_cross_correlation output format
    return [shifts]


def is_3D(dask_array: da.array) -> bool:
    """
    Check if a dask array is 3D.

    Treats singelton Z dimensions as 2D images.
    (1, 2000, 2000) => False
    (10, 2000, 2000) => True

    Args:
        dask_array: Input array to be checked

    Returns:
        bool on whether the array is 3D
    """
    if len(dask_array.shape) == 3 and dask_array.shape[0] > 1:
        return True
    else:
        return False

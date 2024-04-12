# Copyright 2024 (C) BioVisionCenter
#
# Original authors:
# Joel Lüthi <joel.luethi@uzh.ch>
#
# This file is part of Fractal
"""Utils functions for registration"""
import copy

import anndata as ad
import numpy as np
import pandas as pd

from fractal_tasks_core.ngff.zarr_utils import load_NgffWellMeta


def _split_well_path_image_path(zarr_url: str) -> tuple[str, str]:
    """
    Returns path to well folder for HCS OME-Zarr `zarr_url`.
    """
    zarr_url = zarr_url.rstrip("/")
    well_path = "/".join(zarr_url.split("/")[:-1])
    img_path = zarr_url.split("/")[-1]
    return well_path, img_path


def create_well_acquisition_dict(
    zarr_urls: list[str],
) -> dict[str, dict[int, str]]:
    """
    Parses zarr_urls & groups them by HCS wells & acquisition

    Generates a dict with keys a unique description of the acquisition
    (e.g. plate + well for HCS plates). The values are dictionaries. The keys
    of the secondary dictionary are the acqusitions, its values the `zarr_url`
    for a given acquisition.

    Args:
        zarr_urls: List of zarr_urls

    Returns:
        image_groups
    """
    image_groups = dict()

    # Dict to cache well-level metadata
    well_metadata = dict()
    for zarr_url in zarr_urls:
        well_path, img_sub_path = _split_well_path_image_path(zarr_url)
        # For the first zarr_url of a well, load the well metadata and
        # initialize the image_groups dict
        if well_path not in image_groups:
            well_meta = load_NgffWellMeta(well_path)
            well_metadata[well_path] = well_meta.well
            image_groups[well_path] = {}

        # For every zarr_url, add it under the well_path & acquisition keys to
        # the image_groups dict
        for image in well_metadata[well_path].images:
            if image.path == img_sub_path:
                if image.acquisition in image_groups[well_path]:
                    raise ValueError(
                        "This task has not been built for OME-Zarr HCS plates"
                        "with multiple images of the same acquisition per well"
                        f". {image.acquisition} is the acquisition for "
                        f"multiple images in {well_path=}."
                    )

                image_groups[well_path][image.acquisition] = zarr_url
    return image_groups


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
            "enter a wrong reference cycle?"
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
        across all cycles.
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

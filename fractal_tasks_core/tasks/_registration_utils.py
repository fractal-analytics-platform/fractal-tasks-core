# Copyright 2024 (C) BioVisionCenter
#
# Original authors:
# Joel Lüthi <joel.luethi@uzh.ch>
#
# This file is part of Fractal
"""Utils functions for registration"""
import anndata as ad
import numpy as np
import pandas as pd
import zarr

from fractal_tasks_core.ngff.specs import Well


def _split_well_path_image_path(zarr_url: str):
    """
    Returns path to well folder for HCS OME-Zarr zarr_url
    """
    zarr_url = zarr_url.rstrip("/")
    well_path = "/".join(zarr_url.split("/")[:-1])
    img_path = zarr_url.split("/")[-1]
    return well_path, img_path


def create_well_acquisition_dict(zarr_urls: list[str]):
    """
    Parses zarr_urls & groups them by HCS wells & acquisition

    Generates a dict with keys a unique description of the acquisition
    (e.g. plate + well for HCS plates). The values are a dictionary. The keys
    of the secondary dictionary are the acqusitions, its values the zarr_url
    for a given acquisition.
    """
    # Dict with keys a unique description of the acquisition (e.g. plate +
    # well for HCS plates). The values are a dictionary. The keys of the
    # secondary dictionary are the acqusitions, its values the zarr_url for
    # a given acquisition
    image_groups = dict()
    # Dict to cache well-level metadata
    well_metadata = dict()
    for zarr_url in zarr_urls:
        well_path, img_sub_path = _split_well_path_image_path(zarr_url)
        # For the first zarr_url of a well, load the well metadata and
        # initialize the image_groups dict
        if well_path not in image_groups:
            image_groups[well_path] = {}
            well_group = zarr.open_group(well_path, mode="r")
            well_metadata[well_path] = Well(**well_group.attrs.asdict())

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
        shifts_physical: shifts in physical units as zyx
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

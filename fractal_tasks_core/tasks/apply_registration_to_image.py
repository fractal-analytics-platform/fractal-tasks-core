# Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Joel Lüthi <joel.luethi@uzh.ch>
#
# This file is part of Fractal and was originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
"""
Calculates translation for 2D image-based registration
"""
import logging
import os
import shutil
import time
from typing import Callable

import anndata as ad
import dask.array as da
import numpy as np
import zarr
from pydantic.decorator import validate_arguments

from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.pyramids import build_pyramid
from fractal_tasks_core.roi import (
    convert_indices_to_regions,
)
from fractal_tasks_core.roi import (
    convert_ROI_table_to_indices,
)
from fractal_tasks_core.roi import is_standard_roi_table
from fractal_tasks_core.roi import load_region
from fractal_tasks_core.tables import write_table
from fractal_tasks_core.utils import _get_table_path_dict

logger = logging.getLogger(__name__)


@validate_arguments
def apply_registration_to_image(
    *,
    # Fractal argument
    zarr_url: str,
    # Task-specific arguments
    registered_roi_table: str,
    reference_cycle: str = "0",
    overwrite_input: bool = True,
):
    """
    Apply registration to images by using a registered ROI table

    This task consists of 4 parts:

    1. Mask all regions in images that are not available in the
    registered ROI table and store each cycle aligned to the
    reference_cycle (by looping over ROIs).
    2. Do the same for all label images.
    3. Copy all tables from the non-aligned image to the aligned image
    (currently only works well if the only tables are well & FOV ROI tables
    (registered and original). Not implemented for measurement tables and
    other ROI tables).
    4. Clean up: Delete the old, non-aligned image and rename the new,
    aligned image to take over its place.

    Parallelization level: image

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        registered_roi_table: Name of the ROI table which has been registered
            and will be applied to mask and shift the images.
            Examples: `registered_FOV_ROI_table` => loop over the field of
            views, `registered_well_ROI_table` => process the whole well as
            one image.
        reference_cycle: Which cycle to register against. Defaults to 0,
            which is the first OME-Zarr image in the well, usually the first
            cycle that was provided
        overwrite_input: Whether the old image data should be replaced with the
            newly registered image data. Currently only implemented for
            `overwrite_input=True`.

    """
    logger.info(zarr_url)
    logger.info(
        f"Running `apply_registration_to_image` on {zarr_url=}, "
        f"{registered_roi_table=} and {reference_cycle=}. "
        f"Using {overwrite_input=}"
    )

    new_zarr_url = "/".join(
        zarr_url.split("/")[:-1] + [zarr_url.split("/")[-1] + "_registered"]
    )
    # TODO: Make this more robust by actually checking for acquisition
    # metadata
    reference_zarr_url = "/".join(zarr_url.split("/")[:-1] + [reference_cycle])

    ROI_table_ref = ad.read_zarr(
        f"{reference_zarr_url}/tables/{registered_roi_table}"
    )
    ROI_table_cycle = ad.read_zarr(f"{zarr_url}/tables/{registered_roi_table}")

    ngff_image_meta = load_NgffImageMeta(zarr_url)
    coarsening_xy = ngff_image_meta.coarsening_xy
    num_levels = ngff_image_meta.num_levels

    ####################
    # Process images
    ####################
    logger.info("Write the registered Zarr image to disk")
    write_registered_zarr(
        zarr_url=zarr_url,
        new_zarr_url=new_zarr_url,
        ROI_table=ROI_table_cycle,
        ROI_table_ref=ROI_table_ref,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        aggregation_function=np.mean,
    )

    ####################
    # Process labels
    ####################
    try:
        labels_group = zarr.open_group(f"{zarr_url}/labels", "r")
        label_list = labels_group.attrs["labels"]
    except (zarr.errors.GroupNotFoundError, KeyError):
        label_list = []

    if label_list:
        logger.info(f"Processing the label images: {label_list}")
        labels_group = zarr.group(f"{new_zarr_url}/labels")
        labels_group.attrs["labels"] = label_list

        for label in label_list:
            write_registered_zarr(
                zarr_url=f"{zarr_url}/labels/{label}",
                new_zarr_url=f"{new_zarr_url}/labels/{label}",
                ROI_table=ROI_table_cycle,
                ROI_table_ref=ROI_table_ref,
                num_levels=num_levels,
                coarsening_xy=coarsening_xy,
                aggregation_function=np.max,
            )

    ####################
    # Copy tables
    # 1. Copy all standard ROI tables from cycle 0.
    # 2. Copy all tables that aren't standard ROI tables from the given cycle
    ####################
    table_dict_reference = _get_table_path_dict(reference_zarr_url)
    table_dict_component = _get_table_path_dict(zarr_url)

    table_dict = {}
    # Define which table should get copied:
    for table in table_dict_reference:
        if is_standard_roi_table(table):
            table_dict[table] = table_dict_reference[table]
    for table in table_dict_component:
        if not is_standard_roi_table(table):
            if reference_zarr_url != zarr_url:
                logger.warning(
                    f"{zarr_url} contained a table that is not a standard "
                    "ROI table. The `Apply Registration To Image task` is "
                    "best used before additional tables are generated. It "
                    f"will copy the {table} from this cycle without applying "
                    f"any transformations. This will work well if {table} "
                    f"contains measurements. But if {table} is a custom ROI "
                    "table coming from another task, the transformation is "
                    "not applied and it will not match with the registered "
                    "image anymore"
                )
            table_dict[table] = table_dict_component[table]

    if table_dict:
        logger.info(f"Processing the tables: {table_dict}")
        new_image_group = zarr.group(new_zarr_url)

        for table in table_dict.keys():
            logger.info(f"Copying table: {table}")
            # Get the relevant metadata of the Zarr table & add it
            # See issue #516 for the need for this workaround
            max_retries = 20
            sleep_time = 5
            current_round = 0
            while current_round < max_retries:
                try:
                    old_table_group = zarr.open_group(
                        table_dict[table], mode="r"
                    )
                    current_round = max_retries
                except zarr.errors.GroupNotFoundError:
                    logger.debug(
                        f"Table {table} not found in attempt {current_round}. "
                        f"Waiting {sleep_time} seconds before trying again."
                    )
                    current_round += 1
                    time.sleep(sleep_time)
            # Write the Zarr table
            curr_table = ad.read_zarr(table_dict[table])
            write_table(
                new_image_group,
                table,
                curr_table,
                table_attrs=old_table_group.attrs.asdict(),
                overwrite=True,
            )

    ####################
    # Clean up Zarr file
    ####################
    if overwrite_input:
        logger.info(
            "Replace original zarr image with the newly created Zarr image"
        )
        # Potential for race conditions: Every cycle reads the
        # reference cycle, but the reference cycle also gets modified
        # See issue #516 for the details
        os.rename(zarr_url, f"{zarr_url}_tmp")
        os.rename(new_zarr_url, zarr_url)
        shutil.rmtree(f"{zarr_url}_tmp")
        image_list_updates = dict(image_list_updates=[dict(zarr_url=zarr_url)])
    else:
        image_list_updates = dict(
            image_list_updates=[dict(zarr_url=new_zarr_url, origin=zarr_url)]
        )

    return image_list_updates


def write_registered_zarr(
    zarr_url: str,
    new_zarr_url: str,
    ROI_table: ad.AnnData,
    ROI_table_ref: ad.AnnData,
    num_levels: int,
    coarsening_xy: int = 2,
    aggregation_function: Callable = np.mean,
):
    """
    Write registered zarr array based on ROI tables

    This function loads the image or label data from a zarr array based on the
    ROI bounding-box coordinates and stores them into a new zarr array.
    The new Zarr array has the same shape as the original array, but will have
    0s where the ROI tables don't specify loading of the image data.
    The ROIs loaded from `list_indices` will be written into the
    `list_indices_ref` position, thus performing translational registration if
    the two lists of ROI indices vary.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be used as
            the basis for the new OME-Zarr image.
        new_zarr_url: Path or url to the new OME-Zarr image to be written
        ROI_table: Fractal ROI table for the component
        ROI_table_ref: Fractal ROI table for the reference cycle
        num_levels: Number of pyramid layers to be created (argument of
            `build_pyramid`).
        coarsening_xy: Coarsening factor between pyramid levels
        aggregation_function: Function to be used when downsampling (argument
            of `build_pyramid`).

    """
    # Read pixel sizes from Zarr attributes
    ngff_image_meta = load_NgffImageMeta(zarr_url)
    pxl_sizes_zyx = ngff_image_meta.get_pixel_sizes_zyx(level=0)

    # Create list of indices for 3D ROIs
    list_indices = convert_ROI_table_to_indices(
        ROI_table,
        level=0,
        coarsening_xy=coarsening_xy,
        full_res_pxl_sizes_zyx=pxl_sizes_zyx,
    )
    list_indices_ref = convert_ROI_table_to_indices(
        ROI_table_ref,
        level=0,
        coarsening_xy=coarsening_xy,
        full_res_pxl_sizes_zyx=pxl_sizes_zyx,
    )

    old_image_group = zarr.open_group(zarr_url, mode="r")
    old_ngff_image_meta = load_NgffImageMeta(zarr_url)
    new_image_group = zarr.group(new_zarr_url)
    new_image_group.attrs.put(old_image_group.attrs.asdict())

    # Loop over all channels. For each channel, write full-res image data.
    data_array = da.from_zarr(old_image_group["0"])
    # Create dask array with 0s of same shape
    new_array = da.zeros_like(data_array)

    # TODO: Add sanity checks on the 2 ROI tables:
    # 1. The number of ROIs need to match
    # 2. The size of the ROIs need to match
    # (otherwise, we can't assign them to the reference regions)
    # ROI_table_ref vs ROI_table_cycle
    for i, roi_indices in enumerate(list_indices):
        reference_region = convert_indices_to_regions(list_indices_ref[i])
        region = convert_indices_to_regions(roi_indices)

        axes_list = old_ngff_image_meta.axes_names

        if axes_list == ["c", "z", "y", "x"]:
            num_channels = data_array.shape[0]
            # Loop over channels
            for ind_ch in range(num_channels):
                idx = tuple(
                    [slice(ind_ch, ind_ch + 1)] + list(reference_region)
                )
                new_array[idx] = load_region(
                    data_zyx=data_array[ind_ch], region=region, compute=False
                )
        elif axes_list == ["z", "y", "x"]:
            new_array[reference_region] = load_region(
                data_zyx=data_array, region=region, compute=False
            )
        elif axes_list == ["c", "y", "x"]:
            # TODO: Implement cyx case (based on looping over xy case)
            raise NotImplementedError(
                "`write_registered_zarr` has not been implemented for "
                f"a zarr with {axes_list=}"
            )
        elif axes_list == ["y", "x"]:
            # TODO: Implement yx case
            raise NotImplementedError(
                "`write_registered_zarr` has not been implemented for "
                f"a zarr with {axes_list=}"
            )
        else:
            raise NotImplementedError(
                "`write_registered_zarr` has not been implemented for "
                f"a zarr with {axes_list=}"
            )

    new_array.to_zarr(
        f"{new_zarr_url}/0",
        overwrite=True,
        dimension_separator="/",
        write_empty_chunks=False,
    )

    # Starting from on-disk highest-resolution data, build and write to
    # disk a pyramid of coarser levels
    build_pyramid(
        zarrurl=new_zarr_url,
        overwrite=True,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        chunksize=data_array.chunksize,
        aggregation_function=aggregation_function,
    )


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=apply_registration_to_image,
        logger_name=logger.name,
    )

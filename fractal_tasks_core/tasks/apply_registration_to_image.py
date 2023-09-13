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
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Sequence

import anndata as ad
import dask.array as da
import numpy as np
import zarr
from anndata._io.specs import write_elem
from pydantic.decorator import validate_arguments

from fractal_tasks_core.lib_pyramid_creation import build_pyramid
from fractal_tasks_core.lib_regions_of_interest import (
    convert_indices_to_regions,
)
from fractal_tasks_core.lib_regions_of_interest import (
    convert_ROI_table_to_indices,
)
from fractal_tasks_core.lib_regions_of_interest import is_standard_roi_table
from fractal_tasks_core.lib_regions_of_interest import load_region
from fractal_tasks_core.lib_zattrs_utils import extract_zyx_pixel_sizes
from fractal_tasks_core.lib_zattrs_utils import get_axes_names
from fractal_tasks_core.lib_zattrs_utils import get_table_path_dict

logger = logging.getLogger(__name__)


@validate_arguments
def apply_registration_to_image(
    *,
    # Fractal arguments
    input_paths: Sequence[str],
    output_path: str,
    component: str,
    metadata: dict[str, Any],
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
        input_paths: List of input paths where the image data is stored as
            OME-Zarrs. Should point to the parent folder containing one or many
            OME-Zarr files, not the actual OME-Zarr file. Example:
            `["/some/path/"]`. This task only supports a single input path.
            (standard argument for Fractal tasks, managed by Fractal server).
        output_path: This parameter is not used by this task.
            (standard argument for Fractal tasks, managed by Fractal server).
        component: Path to the OME-Zarr image in the OME-Zarr plate that is
            processed. Example: `"some_plate.zarr/B/03/0"`.
            (standard argument for Fractal tasks, managed by Fractal server).
        metadata: Dictionary containing metadata about the OME-Zarr. This task
            requires the following elements to be present in the metadata.
            `coarsening_xy (int)`: coarsening factor in XY of the downsampling
            when building the pyramid. (standard argument for Fractal tasks,
            managed by Fractal server).
            `num_levels (int)`: number of pyramid levels in the image; this
            determines how many pyramid levels are built for the segmentation.
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
    logger.info(component)
    if not overwrite_input:
        raise NotImplementedError(
            "This task is only implemented for the overwrite_input version"
        )
    logger.info(
        f"Running `apply_registration_to_image` on {input_paths=}, "
        f"{component=}, {registered_roi_table=} and {reference_cycle=}. "
        f"Using {overwrite_input=}"
    )
    coarsening_xy = metadata["coarsening_xy"]
    num_levels = metadata["num_levels"]
    input_path = Path(input_paths[0])
    new_component = "/".join(
        component.split("/")[:-1] + [component.split("/")[-1] + "_registered"]
    )
    reference_component = "/".join(
        component.split("/")[:-1] + [reference_cycle]
    )

    ROI_table_ref = ad.read_zarr(
        f"{input_path / reference_component}/tables/{registered_roi_table}"
    )
    ROI_table_cycle = ad.read_zarr(
        f"{input_path / component}/tables/{registered_roi_table}"
    )

    ####################
    # Process images
    ####################
    logger.info("Write the registered Zarr image to disk")
    write_registered_zarr(
        input_path=input_path,
        component=component,
        new_component=new_component,
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
        with open(f"{input_path / component}/labels/.zattrs", "r") as f_zattrs:
            label_list = json.load(f_zattrs)["labels"]
    except FileNotFoundError:
        label_list = []

    if label_list:
        logger.info(f"Processing the label images: {label_list}")
        labels_group = zarr.group(f"{input_path / new_component}/labels")
        labels_group.attrs["labels"] = label_list

        for label in label_list:
            label_component = f"{component}/labels/{label}"
            label_component_new = f"{new_component}/labels/{label}"
            write_registered_zarr(
                input_path=input_path,
                component=label_component,
                new_component=label_component_new,
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
    table_dict_reference = get_table_path_dict(input_path, reference_component)
    table_dict_component = get_table_path_dict(input_path, component)

    table_dict = {}
    # Define which table should get copied:
    for table in table_dict_reference:
        if is_standard_roi_table(table):
            table_dict[table] = table_dict_reference[table]
    for table in table_dict_component:
        if not is_standard_roi_table(table):
            if reference_component != component:
                logger.warning(
                    f"{component} contained a table that is not a standard "
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
        new_tables_group = zarr.group(f"{input_path / new_component}/tables")
        new_tables_group.attrs["tables"] = list(table_dict.keys())

        for table in table_dict.keys():
            logger.info(f"Copying table: {table}")
            # Write the Zarr table
            curr_table = ad.read_zarr(table_dict[table])
            write_elem(new_tables_group, table, curr_table)
            # Get the relevant metadata of the Zarr table & add it
            # See issue #516 for the need for this workaround
            try:
                old_table_group = zarr.open_group(table_dict[table], mode="r")
            except zarr.errors.GroupNotFoundError:
                time.sleep(5)
                old_table_group = zarr.open_group(table_dict[table], mode="r")
            new_table_group = zarr.open_group(
                f"{input_path / new_component}/tables/{table}"
            )
            new_table_group.attrs.put(old_table_group.attrs.asdict())

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
        os.rename(f"{input_path / component}", f"{input_path / component}_tmp")
        os.rename(f"{input_path / new_component}", f"{input_path / component}")
        shutil.rmtree(f"{input_path / component}_tmp")
    else:
        raise NotImplementedError
        # The thing that would be missing in this branch is that Fractal
        # isn't aware of the new component. If there's a way to add it back,
        # that's the only thing that would be required here


def write_registered_zarr(
    input_path: Path,
    component: str,
    new_component: str,
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
        input_path: Base folder where the Zarr is stored
            (does not contain the Zarr file itself)
        component: Path to the OME-Zarr image that is processed. For example:
            `"20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr/B/03/1"`
        new_component: Path to the new Zarr image that will be written
            (also in the input_path folder). For example:
            `"20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr/B/03/1_registered"`
        ROI_table: Fractal ROI table for the component
        ROI_table_ref: Fractal ROI table for the reference cycle
        num_levels: Number of pyramid layers to be created (argument of
            `build_pyramid`).
        coarsening_xy: Coarsening factor between pyramid levels
        aggregation_function: Function to be used when downsampling (argument
            of `build_pyramid`).

    """
    # Read pixel sizes from zattrs file
    pxl_sizes_zyx = extract_zyx_pixel_sizes(
        f"{str(input_path / component)}/.zattrs", level=0
    )

    # Create list of indices for 3D ROIs
    list_indices = convert_ROI_table_to_indices(
        ROI_table,
        level=0,
        coarsening_xy=coarsening_xy,
        full_res_pxl_sizes_zyx=pxl_sizes_zyx,
        reset_origin=False,
    )
    list_indices_ref = convert_ROI_table_to_indices(
        ROI_table_ref,
        level=0,
        coarsening_xy=coarsening_xy,
        full_res_pxl_sizes_zyx=pxl_sizes_zyx,
        reset_origin=False,
    )

    old_image_group = zarr.open_group(f"{input_path / component}", mode="r")
    new_image_group = zarr.group(f"{input_path / new_component}")
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

        axes_list = get_axes_names(old_image_group.attrs.asdict())

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
        f"{input_path / new_component}/0",
        overwrite=True,
        dimension_separator="/",
        write_empty_chunks=False,
    )

    # Starting from on-disk highest-resolution data, build and write to
    # disk a pyramid of coarser levels
    build_pyramid(
        zarrurl=f"{input_path / new_component}",
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

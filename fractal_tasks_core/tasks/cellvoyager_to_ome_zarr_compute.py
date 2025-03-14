# Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Tommaso Comparin <tommaso.comparin@exact-lab.it>
# Marco Franzon <marco.franzon@exact-lab.it>
#
# This file is part of Fractal and was originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
"""
Task that writes image data to an existing OME-NGFF zarr array.
"""
import logging

import dask.array as da
import zarr
from anndata import read_zarr
from dask.array.image import imread
from pydantic import Field
from pydantic import validate_call

from fractal_tasks_core.cellvoyager.filenames import (
    glob_with_multiple_patterns,
)
from fractal_tasks_core.cellvoyager.filenames import parse_filename
from fractal_tasks_core.channels import get_omero_channel_list
from fractal_tasks_core.channels import OmeroChannel
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.pyramids import build_pyramid
from fractal_tasks_core.roi import check_valid_ROI_indices
from fractal_tasks_core.roi import (
    convert_ROI_table_to_indices,
)
from fractal_tasks_core.tasks.io_models import ChunkSizes
from fractal_tasks_core.tasks.io_models import InitArgsCellVoyager


logger = logging.getLogger(__name__)


def sort_fun(filename: str) -> list[int]:
    """
    Takes a string (filename of a Yokogawa image), extract site and
    z-index metadata and returns them as a list of integers.

    Args:
        filename: Name of the image file.
    """

    filename_metadata = parse_filename(filename)
    site = int(filename_metadata["F"])
    z_index = int(filename_metadata["Z"])
    return [site, z_index]


@validate_call
def cellvoyager_to_ome_zarr_compute(
    *,
    # Fractal parameters
    zarr_url: str,
    init_args: InitArgsCellVoyager,
    chunk_sizes: ChunkSizes = Field(default_factory=ChunkSizes),
):
    """
    Convert Yokogawa output (png, tif) to zarr file.

    This task is run after an init task (typically
    `cellvoyager_to_ome_zarr_init` or
    `cellvoyager_to_ome_zarr_init_multiplex`), and it populates the empty
    OME-Zarr files that were prepared.

    Note that the current task always overwrites existing data. To avoid this
    behavior, set the `overwrite` argument of the init task to `False`.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        init_args: Intialization arguments provided by
            `create_cellvoyager_ome_zarr_init`.
        chunk_sizes: Used to overwrite the default chunk sizes for the
            OME-Zarr. By default, the task will chunk the same as the
            microscope field of view size, with 10 z planes per chunk.
            For example, that can mean c: 1, z: 10, y: 2160, x:2560
    """
    zarr_url = zarr_url.rstrip("/")
    # Read attributes from NGFF metadata
    ngff_image_meta = load_NgffImageMeta(zarr_url)
    num_levels = ngff_image_meta.num_levels
    coarsening_xy = ngff_image_meta.coarsening_xy
    full_res_pxl_sizes_zyx = ngff_image_meta.get_pixel_sizes_zyx(level=0)
    logger.info(f"NGFF image has {num_levels=}")
    logger.info(f"NGFF image has {coarsening_xy=}")
    logger.info(
        f"NGFF image has full-res pixel sizes {full_res_pxl_sizes_zyx}"
    )

    channels: list[OmeroChannel] = get_omero_channel_list(
        image_zarr_path=zarr_url
    )
    wavelength_ids = [c.wavelength_id for c in channels]

    # Read useful information from ROI table
    adata = read_zarr(f"{zarr_url}/tables/FOV_ROI_table")
    fov_indices = convert_ROI_table_to_indices(
        adata,
        full_res_pxl_sizes_zyx=full_res_pxl_sizes_zyx,
    )
    check_valid_ROI_indices(fov_indices, "FOV_ROI_table")
    adata_well = read_zarr(f"{zarr_url}/tables/well_ROI_table")
    well_indices = convert_ROI_table_to_indices(
        adata_well,
        full_res_pxl_sizes_zyx=full_res_pxl_sizes_zyx,
    )
    check_valid_ROI_indices(well_indices, "well_ROI_table")
    if len(well_indices) > 1:
        raise ValueError(f"Something wrong with {well_indices=}")

    max_z = well_indices[0][1]
    max_y = well_indices[0][3]
    max_x = well_indices[0][5]

    # Load a single image, to retrieve useful information
    include_patterns = [
        f"{init_args.plate_prefix}_{init_args.well_ID}_*."
        f"{init_args.image_extension}"
    ]
    if init_args.include_glob_patterns:
        include_patterns.extend(init_args.include_glob_patterns)

    exclude_patterns = []
    if init_args.exclude_glob_patterns:
        exclude_patterns.extend(init_args.exclude_glob_patterns)

    tmp_images = glob_with_multiple_patterns(
        folder=init_args.image_dir,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
    )
    sample = imread(tmp_images.pop())

    # Initialize zarr
    chunksize_default = {
        "c": 1,
        "z": 10,
        "y": sample.shape[1],
        "x": sample.shape[2],
    }
    chunksize = chunk_sizes.get_chunksize(chunksize_default=chunksize_default)
    # chunksize["z"] =
    canvas_zarr = zarr.create(
        shape=(len(wavelength_ids), max_z, max_y, max_x),
        chunks=chunksize,
        dtype=sample.dtype,
        store=zarr.storage.FSStore(zarr_url + "/0"),
        overwrite=True,
        dimension_separator="/",
    )

    # Loop over channels
    for i_c, wavelength_id in enumerate(wavelength_ids):
        A, C = wavelength_id.split("_")

        include_patterns = [
            f"{init_args.plate_prefix}_{init_args.well_ID}_*{A}*{C}*."
            f"{init_args.image_extension}"
        ]
        if init_args.include_glob_patterns:
            include_patterns.extend(init_args.include_glob_patterns)
        filenames_set = glob_with_multiple_patterns(
            folder=init_args.image_dir,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )
        filenames = sorted(list(filenames_set), key=sort_fun)
        if len(filenames) == 0:
            raise ValueError(
                "Error in yokogawa_to_ome_zarr: len(filenames)=0.\n"
                f"  image_dir: {init_args.image_dir}\n"
                f"  wavelength_id: {wavelength_id},\n"
                f"  patterns: {include_patterns}\n"
                f"  exclusion patterns: {exclude_patterns}\n"
            )
        # Loop over 3D FOV ROIs
        for indices in fov_indices:
            s_z, e_z, s_y, e_y, s_x, e_x = indices[:]
            region = (
                slice(i_c, i_c + 1),
                slice(s_z, e_z),
                slice(s_y, e_y),
                slice(s_x, e_x),
            )
            FOV_3D = da.concatenate(
                [imread(img) for img in filenames[:e_z]],
            )
            FOV_4D = da.expand_dims(FOV_3D, axis=0)
            filenames = filenames[e_z:]
            da.array(FOV_4D).to_zarr(
                url=canvas_zarr,
                region=region,
                compute=True,
            )

    # Starting from on-disk highest-resolution data, build and write to disk a
    # pyramid of coarser levels
    build_pyramid(
        zarrurl=zarr_url,
        overwrite=True,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        chunksize=chunksize,
    )

    # Generate image list updates
    # TODO: Can we check for dimensionality more robustly? Just checks for the
    # last FOV of the last wavelength now
    if FOV_4D.shape[-3] > 1:
        is_3D = True
    else:
        is_3D = False
    # FIXME: Get plate name from zarr_url => works for duplicate plate names
    # with suffixes
    print(zarr_url)
    plate_name = zarr_url.split("/")[-4]
    attributes = {
        "plate": plate_name,
        "well": init_args.well_ID,
    }
    if init_args.acquisition is not None:
        attributes["acquisition"] = init_args.acquisition

    image_list_updates = dict(
        image_list_updates=[
            dict(
                zarr_url=zarr_url,
                attributes=attributes,
                types={"is_3D": is_3D},
            )
        ]
    )

    return image_list_updates


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=cellvoyager_to_ome_zarr_compute,
        logger_name=logger.name,
    )

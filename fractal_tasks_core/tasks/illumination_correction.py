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
Apply illumination correction to all fields of view.
"""
import logging
import time
import warnings
from pathlib import Path
from typing import Any

import anndata as ad
import dask.array as da
import numpy as np
import zarr
from pydantic import validate_call
from skimage.io import imread

from fractal_tasks_core.channels import get_omero_channel_list
from fractal_tasks_core.channels import OmeroChannel
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.pyramids import build_pyramid
from fractal_tasks_core.roi import check_valid_ROI_indices
from fractal_tasks_core.roi import (
    convert_ROI_table_to_indices,
)
from fractal_tasks_core.tasks._zarr_utils import _copy_hcs_ome_zarr_metadata
from fractal_tasks_core.tasks._zarr_utils import _copy_tables_from_zarr_url

logger = logging.getLogger(__name__)


def correct(
    img_stack: np.ndarray,
    corr_img: np.ndarray,
    background: int = 110,
):
    """
    Corrects a stack of images, using a given illumination profile (e.g. bright
    in the center of the image, dim outside).

    Args:
        img_stack: 4D numpy array (czyx), with dummy size along c.
        corr_img: 2D numpy array (yx)
        background: Background value that is subtracted from the image before
            the illumination correction is applied.
    """

    logger.info(f"Start correct, {img_stack.shape}")

    # Check shapes
    if corr_img.shape != img_stack.shape[2:] or img_stack.shape[0] != 1:
        raise ValueError(
            "Error in illumination_correction:\n"
            f"{img_stack.shape=}\n{corr_img.shape=}"
        )

    # Store info about dtype
    dtype = img_stack.dtype
    dtype_max = np.iinfo(dtype).max

    # Background subtraction
    img_stack[img_stack <= background] = 0
    img_stack[img_stack > background] -= background

    #  Apply the normalized correction matrix (requires a float array)
    # img_stack = img_stack.astype(np.float64)
    new_img_stack = img_stack / (corr_img / np.max(corr_img))[None, None, :, :]

    # Handle edge case: corrected image may have values beyond the limit of
    # the encoding, e.g. beyond 65535 for 16bit images. This clips values
    # that surpass this limit and triggers a warning
    if np.sum(new_img_stack > dtype_max) > 0:
        warnings.warn(
            "Illumination correction created values beyond the max range of "
            f"the current image type. These have been clipped to {dtype_max=}."
        )
        new_img_stack[new_img_stack > dtype_max] = dtype_max

    logger.info("End correct")

    # Cast back to original dtype and return
    return new_img_stack.astype(dtype)


@validate_call
def illumination_correction(
    *,
    # Fractal parameters
    zarr_url: str,
    # Core parameters
    illumination_profiles_folder: str,
    illumination_profiles: dict[str, str],
    background: int = 0,
    input_ROI_table: str = "FOV_ROI_table",
    overwrite_input: bool = True,
    # Advanced parameters
    suffix: str = "_illum_corr",
) -> dict[str, Any]:
    """
    Applies illumination correction to the images in the OME-Zarr.

    Assumes that the illumination correction profiles were generated before
    separately and that the same background subtraction was used during
    calculation of the illumination correction (otherwise, it will not work
    well & the correction may only be partial).

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        illumination_profiles_folder: Path of folder of illumination profiles.
        illumination_profiles: Dictionary where keys match the `wavelength_id`
            attributes of existing channels (e.g.  `A01_C01` ) and values are
            the filenames of the corresponding illumination profiles.
        background: Background value that is subtracted from the image before
            the illumination correction is applied. Set it to `0` if you don't
            want any background subtraction.
        input_ROI_table: Name of the ROI table that contains the information
            about the location of the individual field of views (FOVs) to
            which the illumination correction shall be applied. Defaults to
            "FOV_ROI_table", the default name Fractal converters give the ROI
            tables that list all FOVs separately. If you generated your
            OME-Zarr with a different converter and used Import OME-Zarr to
            generate the ROI tables, `image_ROI_table` is the right choice if
            you only have 1 FOV per Zarr image and `grid_ROI_table` if you
            have multiple FOVs per Zarr image and set the right grid options
            during import.
        overwrite_input: If `True`, the results of this task will overwrite
            the input image data. If false, a new image is generated and the
            illumination corrected data is saved there.
        suffix: What suffix to append to the illumination corrected images.
            Only relevant if `overwrite_input=False`.
    """

    # Defione old/new zarrurls
    if overwrite_input:
        zarr_url_new = zarr_url.rstrip("/")
    else:
        zarr_url_new = zarr_url.rstrip("/") + suffix

    t_start = time.perf_counter()
    logger.info("Start illumination_correction")
    logger.info(f"  {overwrite_input=}")
    logger.info(f"  {zarr_url=}")
    logger.info(f"  {zarr_url_new=}")

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

    # Read channels from .zattrs
    channels: list[OmeroChannel] = get_omero_channel_list(
        image_zarr_path=zarr_url
    )
    num_channels = len(channels)

    # Read FOV ROIs
    FOV_ROI_table = ad.read_zarr(f"{zarr_url}/tables/{input_ROI_table}")

    # Create list of indices for 3D FOVs spanning the entire Z direction
    list_indices = convert_ROI_table_to_indices(
        FOV_ROI_table,
        level=0,
        coarsening_xy=coarsening_xy,
        full_res_pxl_sizes_zyx=full_res_pxl_sizes_zyx,
    )
    check_valid_ROI_indices(list_indices, input_ROI_table)

    # Extract image size from FOV-ROI indices. Note: this works at level=0,
    # where FOVs should all be of the exact same size (in pixels)
    ref_img_size = None
    for indices in list_indices:
        img_size = (indices[3] - indices[2], indices[5] - indices[4])
        if ref_img_size is None:
            ref_img_size = img_size
        else:
            if img_size != ref_img_size:
                raise ValueError(
                    "ERROR: inconsistent image sizes in list_indices"
                )
    img_size_y, img_size_x = img_size[:]

    # Assemble dictionary of matrices and check their shapes
    corrections = {}
    for channel in channels:
        wavelength_id = channel.wavelength_id
        corrections[wavelength_id] = imread(
            (
                Path(illumination_profiles_folder)
                / illumination_profiles[wavelength_id]
            ).as_posix()
        )
        if corrections[wavelength_id].shape != (img_size_y, img_size_x):
            raise ValueError(
                "Error in illumination_correction, "
                "correction matrix has wrong shape."
            )

    # Lazily load highest-res level from original zarr array
    data_czyx = da.from_zarr(f"{zarr_url}/0")

    # Create zarr for output
    if overwrite_input:
        new_zarr = zarr.open(f"{zarr_url_new}/0")
    else:
        new_zarr = zarr.create(
            shape=data_czyx.shape,
            chunks=data_czyx.chunksize,
            dtype=data_czyx.dtype,
            store=zarr.storage.FSStore(f"{zarr_url_new}/0"),
            overwrite=False,
            dimension_separator="/",
        )
        _copy_hcs_ome_zarr_metadata(zarr_url, zarr_url_new)
        # Copy ROI tables from the old zarr_url to keep ROI tables and other
        # tables available in the new Zarr
        _copy_tables_from_zarr_url(zarr_url, zarr_url_new)

    # Iterate over FOV ROIs
    num_ROIs = len(list_indices)
    for i_c, channel in enumerate(channels):
        for i_ROI, indices in enumerate(list_indices):
            # Define region
            s_z, e_z, s_y, e_y, s_x, e_x = indices[:]
            region = (
                slice(i_c, i_c + 1),
                slice(s_z, e_z),
                slice(s_y, e_y),
                slice(s_x, e_x),
            )
            logger.info(
                f"Now processing ROI {i_ROI+1}/{num_ROIs} "
                f"for channel {i_c+1}/{num_channels}"
            )
            # Execute illumination correction
            corrected_fov = correct(
                data_czyx[region].compute(),
                corrections[channel.wavelength_id],
                background=background,
            )
            # Write to disk
            da.array(corrected_fov).to_zarr(
                url=new_zarr,
                region=region,
                compute=True,
            )

    # Starting from on-disk highest-resolution data, build and write to disk a
    # pyramid of coarser levels
    build_pyramid(
        zarrurl=zarr_url_new,
        overwrite=True,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        chunksize=data_czyx.chunksize,
    )

    t_end = time.perf_counter()
    logger.info(f"End illumination_correction, elapsed: {t_end-t_start}")

    if overwrite_input:
        image_list_updates = dict(image_list_updates=[dict(zarr_url=zarr_url)])
    else:
        image_list_updates = dict(
            image_list_updates=[dict(zarr_url=zarr_url_new, origin=zarr_url)]
        )
    return image_list_updates


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=illumination_correction,
        logger_name=logger.name,
    )

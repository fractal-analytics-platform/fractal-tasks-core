"""
Copyright 2022 (C)
    Friedrich Miescher Institute for Biomedical Research and
    University of Zurich

    Original authors:
    Tommaso Comparin <tommaso.comparin@exact-lab.it>
    Marco Franzon <marco.franzon@exact-lab.it>

    This file is part of Fractal and was originally developed by eXact lab
    S.r.l.  <exact-lab.it> under contract with Liberali Lab from the Friedrich
    Miescher Institute for Biomedical Research and Pelkmans Lab from the
    University of Zurich.

Apply illumination correction to all fields of view
"""
import logging
import time
import warnings
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from typing import Sequence

import anndata as ad
import dask.array as da
import numpy as np
import zarr
from pydantic.decorator import validate_arguments
from skimage.io import imread

from fractal_tasks_core.lib_channels import get_omero_channel_list
from fractal_tasks_core.lib_channels import OmeroChannel
from fractal_tasks_core.lib_pyramid_creation import build_pyramid
from fractal_tasks_core.lib_regions_of_interest import (
    convert_ROI_table_to_indices,
)
from fractal_tasks_core.lib_zattrs_utils import extract_zyx_pixel_sizes

logger = logging.getLogger(__name__)


def correct(
    img_stack: np.ndarray,
    corr_img: np.ndarray,
    background: int = 110,
):
    """
    Corrects a stack of images, using a given illumination profile (e.g. bright
    in the center of the image, dim outside).

    img_stack is a four-dimensional (czyx) numpy array, with dummy size along c

    :param img_stack: TBD
    :param corr_img: TBD
    :param background: TBD
    """

    logger.info(f"Start correct, {img_stack.shape}")

    # Check shapes
    if corr_img.shape != img_stack.shape[2:] or img_stack.shape[0] != 1:
        raise Exception(
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


@validate_arguments
def illumination_correction(
    *,
    # Standard arguments
    input_paths: Sequence[str],
    output_path: str,
    component: str,
    metadata: Dict[str, Any],
    # Task-specific arguments
    illumination_profiles_folder: str,
    dict_corr: dict[str, str],
    background: int = 110,
    new_component: Optional[str] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:

    """
    Applies illumination correction to the images in the OME-Zarr

    :param input_paths: List of input paths where the image data is stored
                        as OME-Zarrs. Should point to the parent folder
                        containing one or many OME-Zarr files, not the
                        actual OME-Zarr file.
                        Example: ["/some/path/"]
                        This task only supports a single input path.
                        (standard argument for Fractal tasks,
                        managed by Fractal server)
    :param output_path: Path were the output of this task is stored.
                        Example: "/some/path/" => puts the new OME-Zarr file
                        in the same folder as the input OME-Zarr file
                        "/some/new_path" => puts the new OME-Zarr file
                        into a new folder at ``/some/new_path``
                        (standard argument for Fractal tasks,
                        managed by Fractal server)
    :param component: Path to the OME-Zarr image in the OME-Zarr plate that
                      is processed.
                      Example: "some_plate.zarr/B/03/0"
                      (standard argument for Fractal tasks,
                      managed by Fractal server)
    :param metadata: dictionary containing metadata about the OME-Zarr.
                     This task requires the following elements to be present
                     in the metadata:
                     "num_levels": int, number of pyramid levels in the image.
                     This determines how many pyramid levels are built
                     for the segmentation.
                     "coarsening_xy": int, coarsening factor in XY of the
                     downsampling when building the pyramid.
                     (standard argument for Fractal tasks,
                     managed by Fractal server)
    :param illumination_profiles_folder: Path of folder of illumination
                                         profiles.
    :param dict_corr: Dictionary where keys match the ``wavelength_id``
                      attributes of existing channels (e.g.  ``A01_C01`` ) and
                      values are the filenames of the corresponding
                      illumination profiles.
    :param background: Background value that is subtracted from the image
                       before the illumination correction is applied. Set it
                       to 0 if you don't want any background subtraction.
    :param overwrite: If True, the results of this task will overwrite the
                      input image data. This task is only implemented for
                      ``overwrite=True`` at the moment.
    :param new_component: Not implemented yet.
                          This is not implemented well in Fractal server at
                          the moment, it's unclear how a user would specify
                          fitting new components. If the results shall not
                          overwrite the input data and the output path is
                          the same as the input path, a new component needs
                          to be provided.
                          Example:
                          myplate_new_name.zarr/B/03/0/
    """

    # Preliminary checks
    if len(input_paths) > 1:
        raise NotImplementedError
    if (overwrite and new_component is not None) or (
        new_component is None and not overwrite
    ):
        raise Exception(f"{overwrite=}, but {new_component=}")

    if not overwrite:
        msg = (
            "We still have to harmonize illumination_correction("
            "overwrite=False) with replicate_zarr_structure(..., "
            "suffix=..)"
        )
        raise NotImplementedError(msg)

    # Read some parameters from metadata
    num_levels = metadata["num_levels"]
    coarsening_xy = metadata["coarsening_xy"]

    # Defione old/new zarrurls
    plate, well = component.split(".zarr/")
    in_path = Path(input_paths[0])
    zarrurl_old = (in_path / component).as_posix()
    if overwrite:
        zarrurl_new = zarrurl_old
    else:
        new_plate, new_well = new_component.split(".zarr/")
        if new_well != well:
            raise Exception(f"{well=}, {new_well=}")
        zarrurl_new = (Path(output_path) / new_component).as_posix()

    t_start = time.perf_counter()
    logger.info("Start illumination_correction")
    logger.info(f"  {overwrite=}")
    logger.info(f"  {zarrurl_old=}")
    logger.info(f"  {zarrurl_new=}")

    # Read channels from .zattrs
    channels: list[OmeroChannel] = get_omero_channel_list(
        image_zarr_path=zarrurl_old
    )
    num_channels = len(channels)

    # Read FOV ROIs
    FOV_ROI_table = ad.read_zarr(f"{zarrurl_old}/tables/FOV_ROI_table")

    # Read pixel sizes from zattrs file
    full_res_pxl_sizes_zyx = extract_zyx_pixel_sizes(
        f"{zarrurl_old}/.zattrs", level=0
    )

    # Create list of indices for 3D FOVs spanning the entire Z direction
    list_indices = convert_ROI_table_to_indices(
        FOV_ROI_table,
        level=0,
        coarsening_xy=coarsening_xy,
        full_res_pxl_sizes_zyx=full_res_pxl_sizes_zyx,
    )

    # Extract image size from FOV-ROI indices. Note: this works at level=0,
    # where FOVs should all be of the exact same size (in pixels)
    ref_img_size = None
    for indices in list_indices:
        img_size = (indices[3] - indices[2], indices[5] - indices[4])
        if ref_img_size is None:
            ref_img_size = img_size
        else:
            if img_size != ref_img_size:
                raise Exception(
                    "ERROR: inconsistent image sizes in list_indices"
                )
    img_size_y, img_size_x = img_size[:]

    # Assemble dictionary of matrices and check their shapes
    corrections = {}
    for channel in channels:
        wavelength_id = channel.wavelength_id
        corrections[wavelength_id] = imread(
            (
                Path(illumination_profiles_folder) / dict_corr[wavelength_id]
            ).as_posix()
        )
        if corrections[wavelength_id].shape != (img_size_y, img_size_x):
            raise Exception(
                "Error in illumination_correction, "
                "correction matrix has wrong shape."
            )

    # Lazily load highest-res level from original zarr array
    data_czyx = da.from_zarr(f"{zarrurl_old}/0")

    # Create zarr for output
    if overwrite:
        fov_path = zarrurl_old
        new_zarr = zarr.open(f"{zarrurl_old}/0")
    else:
        fov_path = zarrurl_new
        new_zarr = zarr.create(
            shape=data_czyx.shape,
            chunks=data_czyx.chunksize,
            dtype=data_czyx.dtype,
            store=zarr.storage.FSStore(f"{zarrurl_new}/0"),
            overwrite=False,
            dimension_separator="/",
        )

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
        zarrurl=fov_path,
        overwrite=overwrite,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        chunksize=data_czyx.chunksize,
    )

    t_end = time.perf_counter()
    logger.info(f"End illumination_correction, elapsed: {t_end-t_start}")

    return {}


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=illumination_correction,
        logger_name=logger.name,
    )

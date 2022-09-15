"""
Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
University of Zurich

Original authors:
Tommaso Comparin <tommaso.comparin@exact-lab.it>
Marco Franzon <marco.franzon@exact-lab.it>

This file is part of Fractal and was originally developed by eXact lab S.r.l.
<exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
Institute for Biomedical Research and Pelkmans Lab from the University of
Zurich.
"""
import copy
import logging
import time
import warnings
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Iterable
from typing import Optional

import anndata as ad
import dask
import dask.array as da
import numpy as np
import zarr
from devtools import debug
from skimage.io import imread

from fractal_tasks_core.lib_regions_of_interest import (
    convert_ROI_table_to_indices,
)
from fractal_tasks_core.lib_zattrs_utils import extract_zyx_pixel_sizes


def correct(
    da_img,
    illum_img=None,
    background=110,
    logger=None,
):
    """
    Corrects single Z level input image using an illumination profile

    The illumination profile image can be uint8 or uint16.
    It needs to follow the illumination profile. e.g. bright in the
    center of the image, dim outside

    :param img: input image to be corrected, either uint8 or uint16
    :type img: np.array
    :param illum_img: correction matrix
    :type illum_img: np.array
    :param background: value for background subtraction (optional, default 110)
    :type background: int

    """

    # logger.info(f"Start correct function on image of shape {img.shape}")

    # FIXME FIXME
    img = da_img.compute()

    # Check shapes
    if illum_img.shape != img.shape[2:]:
        raise Exception(
            "Error in illumination_correction\n"
            f"img.shape: {img.shape}\n"
            f"illum_img.shape: {illum_img.shape}"
        )

    # Background subtraction
    # FIXME: is there a problem with these changes?
    # devdoc.net/python/dask-2.23.0-doc/delayed-best-practices.html
    # ?highlight=delayed#don-t-mutate-inputs

    below_threshold = img <= background
    above_threshold = np.logical_not(below_threshold)
    img[below_threshold] = 0
    img[above_threshold] = img[above_threshold] - background
    # FIXME FIXME

    # Apply the illumination correction
    # (normalized by the max value in the illum_img)
    img_corr = img / (illum_img / np.max(illum_img))[None, None, :, :]

    # Handle edge case: The illumination correction can increase a value
    # beyond the limit of the encoding, e.g. beyond 65535 for 16bit
    # images. This clips values that surpass this limit and triggers
    # a warning
    if np.sum(img_corr > np.iinfo(img.dtype).max) > 0:
        warnings.warn(
            f"The illumination correction created values \
                       beyond the max range of your current image \
                       type. These have been clipped to \
                       {np.iinfo(img.dtype).max}"
        )
        img_corr[img_corr > np.iinfo(img.dtype).max] = np.iinfo(img.dtype).max

    # logger.info("End correct function")

    # FIXME FIXME
    return da.array(img_corr.astype(img.dtype))


def illumination_correction(
    *,
    input_paths: Iterable[Path],
    output_path: Path,
    metadata: Optional[Dict[str, Any]] = None,
    component: str = None,
    overwrite: bool = False,
    dict_corr: dict = None,
    background: int = 100,
    logger: logging.Logger = None,
):

    """
    FIXME

    Example inputs:
    input_paths: [PosixPath('tmp_out/*.zarr')]
    output_path: PosixPath('tmp_out/*.zarr')
    component: myplate.zarr/B/03/0/
    metadata: {...}
    """

    if logger is None:
        logger = logging.getLogger(__name__)

    # Read some parameters from metadata
    coarsening_xy = metadata["coarsening_xy"]
    chl_list = metadata["channel_list"]
    plate, well = component.split(".zarr/")

    if overwrite:
        raise NotImplementedError("Only overwrite=False currently supported")

    # Define zarrurl
    if len(input_paths) > 1:
        raise NotImplementedError
    in_path = input_paths[0]
    zarrurl = (in_path.parent.resolve() / component).as_posix() + "/"

    # Sanitize zarr paths
    # FIXME

    newzarrurl = zarrurl.replace(".zarr", "_CORR.zarr").replace(
        str(in_path.parent), str(output_path.parent)
    )
    debug(newzarrurl)

    t_start = time.perf_counter()
    logger.info("Start illumination_correction")
    logger.info(f"  {zarrurl=}")
    logger.info(f"  {newzarrurl=}")

    # Read FOV ROIs
    FOV_ROI_table = ad.read_zarr(f"{zarrurl}tables/FOV_ROI_table")

    # Read pixel sizes from zattrs file
    full_res_pxl_sizes_zyx = extract_zyx_pixel_sizes(
        zarrurl + ".zattrs", level=0
    )

    # Create list of indices for 3D FOVs spanning the entire Z direction
    list_indices = convert_ROI_table_to_indices(
        FOV_ROI_table,
        level=0,
        coarsening_xy=coarsening_xy,
        full_res_pxl_sizes_zyx=full_res_pxl_sizes_zyx,
    )

    # Extract image size from FOV-ROI indices
    # Note: this works at level=0, where FOVs should all be of the exact same
    #       size (in pixels)
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

    # Load paths of correction matrices
    root_path_corr = dict_corr.pop("root_path_corr")
    if not root_path_corr.endswith("/"):
        root_path_corr += "/"

    # Assemble dictionary of matrices and check their shapes
    corrections = {}
    for ind_ch, ch in enumerate(chl_list):
        corrections[ch] = imread(root_path_corr + dict_corr[ch])
        if corrections[ch].shape != (img_size_y, img_size_x):
            raise Exception(
                "Error in illumination_correction, "
                "correction matrix has wrong shape."
            )

    # Load highest-resolution level from original zarr array
    data_czyx = da.from_zarr(zarrurl + "/0")
    dtype = data_czyx.dtype
    n_c, n_z, n_y, n_x = data_czyx.shape[:]

    new_zarr = zarr.create(
        shape=data_czyx.shape,
        chunks=data_czyx.chunksize,
        dtype=data_czyx.dtype,
        store=da.core.get_mapper(newzarrurl + "/0"),
        overwrite=False,
        dimension_separator="/",
    )

    regions = []
    for i_c, channel in enumerate(chl_list):
        for indices in list_indices:
            s_z, e_z, s_y, e_y, s_x, e_x = indices[:]
            for i_z in range(s_z, e_z):
                region = (
                    slice(i_c, i_c + 1),
                    slice(i_z, i_z + 1),
                    slice(s_y, e_y),
                    slice(s_x, e_x),
                )
                regions.append(region)

    for region in regions:
        corrected_img = correct(
            data_czyx[region],
            corrections[channel],
            background=background,
            logger=logger,
        )
        task = corrected_img.to_zarr(
            url=new_zarr, region=region, compute=True, dimension_separator="/"
        )

    t_end = time.perf_counter()
    logger.info(f"End illumination_correction, elapsed: {t_end-t_start}")


if __name__ == "__main__":

    # FIXME
    raise NotImplementedError("TODO: CLI argument parsing is not up to date")

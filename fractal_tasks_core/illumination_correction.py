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
import json
import logging
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
from skimage.io import imread

from fractal_tasks_core.lib_pyramid_creation import write_pyramid
from fractal_tasks_core.lib_regions_of_interest import (
    convert_ROI_table_to_indices,
)
from fractal_tasks_core.lib_zattrs_utils import extract_zyx_pixel_sizes


def correct(
    img,
    illum_img=None,
    background=110,
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

    logging.info(f"Start correct function on image of shape {img.shape}")

    # Check shapes
    if illum_img.shape != img.shape:
        raise Exception(
            "Error in illumination_correction\n"
            f"img.shape: {img.shape}\n"
            f"illum_img.shape: {illum_img.shape}"
        )

    # Background subtraction
    # FIXME: is there a problem with these changes?
    # devdoc.net/python/dask-2.23.0-doc/delayed-best-practices.html
    # ?highlight=delayed#don-t-mutate-inputs
    img[img <= background] = 0
    img[img > background] = img[img > background] - background

    # Apply the illumination correction
    # (normalized by the max value in the illum_img)
    img_corr = img / (illum_img / np.max(illum_img))

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

    logging.info("End correct function")

    return img_corr.astype(img.dtype)


def illumination_correction(
    *,
    input_paths: Iterable[Path],
    output_path: Path,
    metadata: Optional[Dict[str, Any]] = None,
    component: str = None,
    overwrite: bool = False,
    dict_corr: dict = None,
    background: int = 100,
):

    """
    FIXME

    Example inputs:
    input_paths: [PosixPath('tmp_out/*.zarr')]
    output_path: PosixPath('tmp_out/*.zarr')
    component: myplate.zarr/B/03/0/
    metadata: {...}
    """

    # Read some parameters from metadata
    num_levels = metadata["num_levels"]
    coarsening_xy = metadata["coarsening_xy"]
    chl_list = metadata["channel_list"]
    plate, well = component.split(".zarr/")

    if not overwrite:
        raise NotImplementedError("Only overwrite=True currently supported")

    # Define zarrurl
    if len(input_paths) > 1:
        raise NotImplementedError
    in_path = input_paths[0]
    zarrurl = (in_path.parent.resolve() / component).as_posix() + "/"

    # Sanitize zarr paths
    newzarrurl = zarrurl

    logging.info(
        "Start illumination_correction " f"with {zarrurl=} and {newzarrurl=}"
    )

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

    # Read number of levels from .zattrs of original zarr file
    with open(zarrurl + ".zattrs", "r") as inputjson:
        zattrs = json.load(inputjson)
    num_levels = len(zattrs["multiscales"][0]["datasets"])

    # Load highest-resolution level from original zarr array
    data_czyx = da.from_zarr(zarrurl + "/0")
    dtype = data_czyx.dtype

    # Check that input array is made of images (in terms of shape/chunks)
    nc, nz, ny, nx = data_czyx.shape
    if (ny % img_size_y != 0) or (nx % img_size_x != 0):
        raise Exception(
            "Error in illumination_correction, "
            f"data_czyx.shape: {data_czyx.shape}"
        )
    chunks_c, chunks_z, chunks_y, chunks_x = data_czyx.chunks
    if len(set(chunks_c)) != 1 or chunks_c[0] != 1:
        raise Exception(
            f"Error in illumination_correction, chunks_c: {chunks_c}"
        )
    if len(set(chunks_z)) != 1 or chunks_z[0] != 1:
        raise Exception(
            f"Error in illumination_correction, chunks_z: {chunks_z}"
        )
    if len(set(chunks_y)) != 1 or chunks_y[0] != img_size_y:
        raise Exception(
            f"Error in illumination_correction, chunks_y: {chunks_y}"
        )
    if len(set(chunks_x)) != 1 or chunks_x[0] != img_size_x:
        raise Exception(
            f"Error in illumination_correction, chunks_x: {chunks_x}"
        )

    # Prepare delayed function
    delayed_correct = dask.delayed(correct)

    # Loop over channels
    data_czyx_new = []
    data_czyx_new = da.empty(
        data_czyx.shape,
        chunks="auto",
        dtype=data_czyx.dtype,
    )
    for ind_ch, ch in enumerate(chl_list):
        # Set correction matrix
        illum_img = corrections[ch]
        # 3D data for multiple FOVs
        # Loop over FOVs
        for indices in list_indices:
            s_z, e_z, s_y, e_y, s_x, e_x = indices[:]
            # 3D single-FOV data
            tmp_zyx = []
            # For each FOV, loop over Z planes
            for ind_z in range(e_z):
                shape = [e_y - s_y, e_x - s_x]
                new_img = delayed_correct(
                    data_czyx[ind_ch, ind_z, s_y:e_y, s_x:e_x],
                    illum_img,
                    background=background,
                )
                tmp_zyx.append(da.from_delayed(new_img, shape, dtype))
            data_czyx_new[ind_ch, s_z:e_z, s_y:e_y, s_x:e_x] = da.stack(
                tmp_zyx, axis=0
            )
    tmp_accumulated_data = data_czyx_new
    accumulated_data = tmp_accumulated_data.rechunk(data_czyx.chunks)

    # Construct resolution pyramid
    write_pyramid(
        accumulated_data,
        newzarrurl=newzarrurl,
        overwrite=overwrite,
        coarsening_xy=coarsening_xy,
        num_levels=num_levels,
        chunk_size_x=img_size_x,
        chunk_size_y=img_size_y,
    )

    logging.info("End illumination_correction")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(prog="illumination_correction.py")
    parser.add_argument(
        "-z", "--zarrurl", help="zarr url, at the FOV level", required=True
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="overwrite original zarr file",
    )
    parser.add_argument(
        "-znew",
        "--newzarrurl",
        help="path of the new zarr file",
    )
    parser.add_argument(
        "--path_dict_corr",
        help="path of JSON file with info on illumination matrices",
    )
    parser.add_argument(
        "-C",
        "--chl_list",
        nargs="+",
        help="list of channel names (e.g. A01_C01)",
    )
    parser.add_argument(
        "-cxy",
        "--coarsening_xy",
        default=2,
        type=int,
        help="coarsening factor along X and Y (optional, defaults to 2)",
    )
    parser.add_argument(
        "-bg",
        "--background",
        default=110,
        type=int,
        help=(
            "threshold for background subtraction"
            " (optional, defaults to 110)"
        ),
    )

    args = parser.parse_args()
    illumination_correction(
        args.zarrurl,
        overwrite=args.overwrite,
        newzarrurl=args.newzarrurl,
        path_dict_corr=args.path_dict_corr,
        chl_list=args.chl_list,
        coarsening_xy=args.coarsening_xy,
        background=args.background,
    )

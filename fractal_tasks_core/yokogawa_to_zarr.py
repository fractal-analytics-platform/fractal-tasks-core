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
import os
import re
from glob import glob
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Iterable
from typing import Optional

import dask.array as da
import zarr
from anndata import read_zarr
from dask.array.image import imread

from fractal_tasks_core.lib_regions_of_interest import (
    convert_ROI_table_to_indices,
)
from fractal_tasks_core.lib_zattrs_utils import extract_zyx_pixel_sizes

# import numpy as np

# from fractal_tasks_core.lib_pyramid_creation import write_pyramid

# from skimage.io import imread


def get_ROIs_bounding_box(adata, pxl_size):
    # x_min_micrometer = min(adata[:, "x_micrometer"].X)
    x_max_micrometer = max(
        adata[:, "x_micrometer"].X + adata[:, "len_x_micrometer"].X
    )
    # ind_x_min = x_min_micrometer / pxl_size[2]
    ind_x_max = x_max_micrometer / pxl_size[2]
    # y_min_micrometer = min(adata[:, "y_micrometer"].X)
    y_max_micrometer = max(
        adata[:, "y_micrometer"].X + adata[:, "len_y_micrometer"].X
    )
    # ind_y_min = y_min_micrometer / pxl_size[1]
    ind_y_max = y_max_micrometer / pxl_size[1]
    # z_min_micrometer = min(adata[:, "z_micrometer"].X)
    z_max_micrometer = max(
        adata[:, "z_micrometer"].X + adata[:, "len_z_micrometer"].X
    )
    # ind_z_min = z_min_micrometer / pxl_size[0]
    ind_z_max = z_max_micrometer / pxl_size[0]

    # assert ind_x_min == 0
    # assert ind_y_min == 0
    # assert ind_z_min == 0

    return ind_x_max, ind_y_max, ind_z_max


def sort_fun(s):
    """
    sort_fun takes a string (filename of a yokogawa images),
    extract site and z-index metadata and returns them as a list.

    :param s: filename
    :type s: str
    """

    site = re.findall(r"F(.*)L", s)[0]
    zind = re.findall(r"Z(.*)C", s)[0]
    return [site, zind]


def yokogawa_to_zarr(
    *,
    input_paths: Iterable[Path],
    output_path: Path,
    delete_input=False,
    metadata: Optional[Dict[str, Any]] = None,
    component: str = None,
):
    """
    Convert Yokogawa output (png, tif) to zarr file

    Example arguments:
      input_paths[0] = /tmp/output/*.zarr  (Path)
      output_path = /tmp/output/*.zarr      (Path)
      metadata = {"channel_list": [...], "num_levels": ..., }
      component = plate.zarr/B/03/0/
    """

    # Preliminary checks
    if len(input_paths) > 1:
        raise NotImplementedError

    chl_list = metadata["channel_list"]
    original_path_list = metadata["original_paths"]
    in_path = Path(original_path_list[0]).parent
    ext = Path(original_path_list[0]).name
    # num_levels = metadata["num_levels"]
    # coarsening_xy = metadata["coarsening_xy"]

    # Hard-coded values (by now) of chunk sizes to be passed to rechunk,
    # both at level 0 (before coarsening) and at levels 1,2,.. (after
    # repeated coarsening).
    # Note that balance=True may override these values.
    # chunk_size_x = 2560
    # chunk_size_y = 2160

    # Define well
    component_split = component.split("/")
    well_row = component_split[1]
    well_column = component_split[2]

    well_ID = well_row + well_column

    # delayed_imread = delayed(imread)
    # from devtools import debug
    # debug(f"Channels: {chl_list}")

    zarrurl = input_paths[0].parent.as_posix() + f"/{component}"
    adata = read_zarr(f"{zarrurl}/tables/FOV_ROI_table")
    pxl_size = extract_zyx_pixel_sizes(f"{zarrurl}/.zattrs")
    fov_indices = convert_ROI_table_to_indices(
        adata, full_res_pxl_sizes_zyx=pxl_size
    )

    max_x, max_y, max_z = get_ROIs_bounding_box(adata, pxl_size)

    # ref_img_size = None
    # for indices in fov_indices:
    #     img_size = (indices[3] - indices[2], indices[5] - indices[4])
    #     if ref_img_size is None:
    #         ref_img_size = img_size
    #     else:
    #         if img_size != ref_img_size:
    #             raise Exception(
    #                 "ERROR: inconsistent image sizes in list_indices"
    #             )

    # img_size_y, img_size_x = img_size[:]

    sample = imread(glob(f"{in_path}/*_{well_ID}_*{ext}")[0])
    from devtools import debug

    debug(f"{sample.shape=}, {sample.shape[1]=}")
    canvas_zarr = zarr.create(
        shape=(len(chl_list), max_z, max_y, max_x),
        chunks=(1, 1, sample.shape[1], sample.shape[2]),
        dtype=sample.dtype,
        store=da.core.get_mapper(zarrurl + "/0"),
        overwrite=False,
        dimension_separator="/",
    )
    # list_channels = []
    for i_c, chl in enumerate(chl_list):
        A, C = chl.split("_")

        glob_path = f"{in_path}/*_{well_ID}_*{A}*{C}{ext}"
        print(f"glob path: {glob_path}")
        filenames = sorted(glob(glob_path), key=sort_fun)
        if len(filenames) == 0:
            raise Exception(
                "Error in yokogawa_to_zarr: len(filenames)=0.\n"
                f"  in_path: {in_path}\n"
                f"  ext: {ext}\n"
                f"  well_ID: {well_ID}\n"
                f"  channel: {chl},\n"
                f"  glob_path: {glob_path}"
            )

        # max_x = max(roi[5] for roi in fov_position)
        # max_y = max(roi[3] for roi in fov_position)
        # max_z = max(roi[1] for roi in fov_position)

        # img_position = []
        # for fov in fov_position:
        #     for z in range(fov[1]):
        #         img = [z, z + 1, fov[2], fov[3], fov[4], fov[5]]
        #         img_position.append(img)

        # regions = []
        #        for i_c, channel in enumerate(chl_list):
        for indices in fov_indices:
            s_z, e_z, s_y, e_y, s_x, e_x = indices[:]
            # for i_z in range(s_z, e_z):
            region = (
                slice(i_c, i_c + 1),
                slice(s_z, e_z),
                slice(s_y, e_y),
                slice(s_x, e_x),
            )

            # assert s_z == 0

            FOV = da.concatenate(
                [imread(img) for img in filenames[:e_z]],
            )
            FOV_4D = da.expand_dims(FOV, axis=0)
            debug(FOV_4D)
            filenames = filenames[e_z:]

            da.array(FOV_4D).to_zarr(
                url=canvas_zarr,
                region=region,
                compute=True,
            )

        # canvas = da.zeros(
        #     (max_z, max_y, max_x),
        #     dtype=sample.dtype,
        #     chunks=(1, chunk_size_y, chunk_size_x),
        # )

    #     for indexes, image_file in zip(*(img_position, filenames)):
    #         canvas[
    #             indexes[0] : indexes[1],  # noqa: 203
    #             indexes[2] : indexes[3],  # noqa: 203
    #             indexes[4] : indexes[5],  # noqa: 203
    #         ] = imread(image_file)

    #     list_channels.append(canvas)
    # data_czyx = da.stack(list_channels, axis=0)

    if delete_input:
        for f in filenames:
            try:
                os.remove(f)
            except OSError as e:
                print("Error: %s : %s" % (f, e.strerror))

    # Construct resolution pyramid
    # write_pyramid(
    #     data_czyx,
    #     newzarrurl=output_path.parent.as_posix() + f"/{component}",
    #     overwrite=False,
    #     coarsening_xy=coarsening_xy,
    #     num_levels=num_levels,
    #     chunk_size_x=chunk_size_x,
    #     chunk_size_y=chunk_size_y,
    # )


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(prog="Yokogawa_to_zarr")

    parser.add_argument(
        "-i", "--in_path", help="directory containing the input files"
    )

    parser.add_argument(
        "-z",
        "--zarrurl",
        help="structure of the zarr folder",
    )

    parser.add_argument(
        "-e",
        "--ext",
        help="source images extension",
    )

    parser.add_argument(
        "-C",
        "--chl_list",
        nargs="+",
        help="list of channel names (e.g. A01_C01)",
    )

    parser.add_argument(
        "-nl",
        "--num_levels",
        type=int,
        help="number of levels in the Zarr pyramid",
    )

    parser.add_argument(
        "-cxy",
        "--coarsening_xy",
        default=2,
        type=int,
        help="coarsening factor along X and Y (optional, defaults to 2)",
    )

    parser.add_argument(
        "-d",
        "--delete_input",
        action="store_true",
        help="Delete input files",
    )

    args = parser.parse_args()

    yokogawa_to_zarr(
        args.zarrurl,
        in_path=args.in_path,
        ext=args.ext,
        chl_list=args.chl_list,
        num_levels=args.num_levels,
        coarsening_xy=args.coarsening_xy,
        delete_input=args.delete_input,
    )

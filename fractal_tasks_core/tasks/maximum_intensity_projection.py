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

Task for 3D->2D maximum-intensity projection
"""
import logging
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Sequence

import anndata as ad
import dask.array as da
from pydantic.decorator import validate_arguments

from fractal_tasks_core.lib_pyramid_creation import build_pyramid
from fractal_tasks_core.lib_regions_of_interest import (
    convert_ROI_table_to_indices,
)
from fractal_tasks_core.lib_zattrs_utils import extract_zyx_pixel_sizes

logger = logging.getLogger(__name__)


@validate_arguments
def maximum_intensity_projection(
    *,
    input_paths: Sequence[str],
    output_path: str,
    component: str,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Perform maximum-intensity projection along Z axis, and store the output in
    a new zarr file.

    :param input_paths: This parameter is not used by this task
                        This task only supports a single input path.
                        (standard argument for Fractal tasks,
                        managed by Fractal server)
    :param output_path: Path were the output of this task is stored.
                        Example: "/some/path/" => puts the new OME-Zarr file
                        in that folder
                        (standard argument for Fractal tasks,
                        managed by Fractal server)
    :param component: Path to the OME-Zarr image in the OME-Zarr plate that
                      is processed. Component is typically changed by the
                      copy_ome_zarr task before to point to a new mip Zarr
                      file.
                      Example: "some_plate_mip.zarr/B/03/0"
                      (standard argument for Fractal tasks,
                      managed by Fractal server)
    :param metadata: dictionary containing metadata about the OME-Zarr.
                     This task requires the following elements to be present
                     in the metadata:
                     "num_levels": int, number of pyramid levels in the image.
                     This determines how many pyramid levels are built for
                     the segmentation.
                     "coarsening_xy": int, coarsening factor in XY of the
                     downsampling when building the pyramid.
                     "plate": List of plates. Example: ["MyPlate.zarr"]
                     "well": List of wells in the OME-Zarr plate.
                     ["MyPlate.zarr/B/03", "MyPlate.zarr/B/05"]
                     "image": List of images in the OME-Zarr plate. Example:
                     ["MyPlate.zarr/B/03/0", "MyPlate.zarr/B/05/0"]
                     (standard argument for Fractal tasks,
                     managed by Fractal server)
    """

    # Preliminary checks
    if len(input_paths) > 1:
        raise NotImplementedError

    # Read some parameters from metadata
    num_levels = metadata["num_levels"]
    coarsening_xy = metadata["coarsening_xy"]
    plate, well = component.split(".zarr/")

    zarrurl_old = metadata["copy_ome_zarr"]["sources"][plate] + "/" + well
    clean_output_path = Path(output_path).resolve()
    zarrurl_new = (clean_output_path / component).as_posix()
    logger.info(f"{zarrurl_old=}")
    logger.info(f"{zarrurl_new=}")

    # This whole block finds (chunk_size_y,chunk_size_x)
    FOV_ROI_table = ad.read_zarr(f"{zarrurl_old}/tables/FOV_ROI_table")
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
    chunk_size_y, chunk_size_x = img_size[:]
    chunksize = (1, 1, chunk_size_y, chunk_size_x)

    # Load 0-th level
    data_czyx = da.from_zarr(zarrurl_old + "/0")
    num_channels = data_czyx.shape[0]
    # Loop over channels
    accumulate_chl = []
    for ind_ch in range(num_channels):
        # Perform MIP for each channel of level 0
        mip_yx = da.stack([da.max(data_czyx[ind_ch], axis=0)], axis=0)
        accumulate_chl.append(mip_yx)
    accumulated_array = da.stack(accumulate_chl, axis=0)

    # Write to disk (triggering execution)
    if accumulated_array.chunksize != chunksize:
        raise Exception("ERROR\n{accumulated_array.chunksize=}\n{chunksize=}")
    accumulated_array.to_zarr(
        f"{zarrurl_new}/0",
        overwrite=False,
        dimension_separator="/",
        write_empty_chunks=False,
    )

    # Starting from on-disk highest-resolution data, build and write to disk a
    # pyramid of coarser levels
    build_pyramid(
        zarrurl=zarrurl_new,
        overwrite=False,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        chunksize=chunksize,
    )

    return {}


if __name__ == "__main__":

    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=maximum_intensity_projection,
        logger_name=logger.name,
    )

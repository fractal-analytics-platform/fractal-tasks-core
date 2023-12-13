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
Task for 3D->2D maximum-intensity projection.
"""
import logging
from pathlib import Path
from typing import Any
from typing import Sequence

import dask.array as da
from pydantic.decorator import validate_arguments
from zarr.errors import ContainsArrayError

from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.pyramids import build_pyramid
from fractal_tasks_core.zarr_utils import OverwriteNotAllowedError

logger = logging.getLogger(__name__)


@validate_arguments
def maximum_intensity_projection(
    *,
    input_paths: Sequence[str],
    output_path: str,
    component: str,
    metadata: dict[str, Any],
    overwrite: bool = False,
) -> dict[str, Any]:
    """
    Perform maximum-intensity projection along Z axis.

    Note: this task stores the output in a new zarr file.

    Args:
        input_paths: This parameter is not used by this task.
            This task only supports a single input path.
            (standard argument for Fractal tasks, managed by Fractal server).
        output_path: Path were the output of this task is stored.
            Example: `"/some/path/"` => puts the new OME-Zarr file in that
            folder.
            (standard argument for Fractal tasks, managed by Fractal server).
        component: Path to the OME-Zarr image in the OME-Zarr plate that
            is processed. Component is typically changed by the `copy_ome_zarr`
            task before, to point to a new mip Zarr file.
            Example: `"some_plate_mip.zarr/B/03/0"`.
            (standard argument for Fractal tasks, managed by Fractal server).
        metadata: Dictionary containing metadata about the OME-Zarr.
            This task requires the key `copy_ome_zarr` to be present in the
            metadata (as defined in `copy_ome_zarr` task).
            (standard argument for Fractal tasks, managed by Fractal server).
        overwrite: If `True`, overwrite the task output.
    """

    # Preliminary checks
    if len(input_paths) > 1:
        raise NotImplementedError

    plate, well = component.split(".zarr/")
    zarrurl_old = metadata["copy_ome_zarr"]["sources"][plate] + "/" + well
    clean_output_path = Path(output_path).resolve()
    zarrurl_new = (clean_output_path / component).as_posix()
    logger.info(f"{zarrurl_old=}")
    logger.info(f"{zarrurl_new=}")

    # Read some parameters from metadata
    ngff_image = load_NgffImageMeta(zarrurl_old)
    num_levels = ngff_image.num_levels
    coarsening_xy = ngff_image.coarsening_xy

    # Load 0-th level
    data_czyx = da.from_zarr(zarrurl_old + "/0")
    num_channels = data_czyx.shape[0]
    chunksize_y = data_czyx.chunksize[-2]
    chunksize_x = data_czyx.chunksize[-1]
    logger.info(f"{num_channels=}")
    logger.info(f"{chunksize_y=}")
    logger.info(f"{chunksize_x=}")
    # Loop over channels
    accumulate_chl = []
    for ind_ch in range(num_channels):
        # Perform MIP for each channel of level 0
        mip_yx = da.stack([da.max(data_czyx[ind_ch], axis=0)], axis=0)
        accumulate_chl.append(mip_yx)
    accumulated_array = da.stack(accumulate_chl, axis=0)

    # Write to disk (triggering execution)
    try:
        accumulated_array.to_zarr(
            f"{zarrurl_new}/0",
            overwrite=overwrite,
            dimension_separator="/",
            write_empty_chunks=False,
        )
    except ContainsArrayError as e:
        error_msg = (
            f"Cannot write array to zarr group at '{zarrurl_new}/0', "
            f"with {overwrite=} (original error: {str(e)}).\n"
            "Hint: try setting overwrite=True."
        )
        logger.error(error_msg)
        raise OverwriteNotAllowedError(error_msg)

    # Starting from on-disk highest-resolution data, build and write to disk a
    # pyramid of coarser levels
    build_pyramid(
        zarrurl=zarrurl_new,
        overwrite=overwrite,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        chunksize=(1, 1, chunksize_y, chunksize_x),
    )

    return {}


if __name__ == "__main__":

    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=maximum_intensity_projection,
        logger_name=logger.name,
    )

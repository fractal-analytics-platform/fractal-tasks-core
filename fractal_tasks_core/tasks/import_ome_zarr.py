# Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Tommaso Comparin <tommaso.comparin@exact-lab.it>
#
# This file is part of Fractal and was originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
"""
Task to import an OME-Zarr.
"""
import logging
import math
from pathlib import Path
from typing import Any
from typing import Literal
from typing import Optional
from typing import Sequence

import anndata as ad
import dask.array as da
import numpy as np
import zarr
from devtools import debug
from pydantic.decorator import validate_arguments

from fractal_tasks_core.lib_ngff import NgffImageMeta
from fractal_tasks_core.lib_write import write_table

logger = logging.getLogger(__name__)


def _get_image_ROI_table(
    array_shape: tuple[int, int, int],
    pixels_ZYX: list[float],
):
    """
    FIXME docstring
    FIXME: make this more flexible, and move it to some ROI module
    """
    shape_z, shape_y, shape_x = array_shape[-3:]
    ROI_table = ad.AnnData(
        X=np.array(
            [
                [
                    0.0,
                    0.0,
                    0.0,
                    shape_x * pixels_ZYX[2],
                    shape_y * pixels_ZYX[1],
                    shape_z * pixels_ZYX[0],
                    0.0,
                    0.0,
                ],
            ],
            dtype=np.float32,
        )
    )
    debug(ROI_table.X)
    ROI_table.obs_names = ["image_1"]
    ROI_table.var_names = [
        "x_micrometer",
        "y_micrometer",
        "z_micrometer",
        "len_x_micrometer",
        "len_y_micrometer",
        "len_z_micrometer",
        "x_micrometer_original",
        "y_micrometer_original",
    ]
    return ROI_table


def _get_grid_ROI_table(
    array_shape: tuple[int, int, int],
    pixels_ZYX: list[float],
    grid_ROI_shape: Optional[tuple[int, int]] = None,
):
    """
    FIXME docstring
    FIXME: make this more flexible, and move it to some ROI module
    """
    shape_z, shape_y, shape_x = array_shape[-3:]
    if grid_ROI_shape is None:
        grid_ROI_shape = (2, 2)
    grid_size_y, grid_size_x = grid_ROI_shape[:]
    X = []
    obs_names = []
    counter = 0
    start_z = 0
    len_z = shape_z

    # Find minimal len_y that covers [0,shape_y] with grid_size_y intervals
    debug(shape_y, shape_x)
    debug(grid_size_y, grid_size_x)
    len_y = math.ceil(shape_y / grid_size_y)
    len_x = math.ceil(shape_x / grid_size_x)
    debug(len_y, len_x)
    for ind_y in range(grid_size_y):
        start_y = ind_y * len_y
        tmp_len_y = min(shape_y, start_y + len_y) - start_y
        for ind_x in range(grid_size_x):
            start_x = ind_x * len_x
            tmp_len_x = min(shape_x, start_x + len_x) - start_x
            X.append(
                [
                    start_x * pixels_ZYX[2],
                    start_y * pixels_ZYX[1],
                    start_z * pixels_ZYX[0],
                    tmp_len_x * pixels_ZYX[2],
                    tmp_len_y * pixels_ZYX[1],
                    len_z * pixels_ZYX[0],
                    start_x * pixels_ZYX[2],
                    start_y * pixels_ZYX[1],
                ]
            )
            debug(X[-1])
            counter += 1
            obs_names.append(f"ROI_{counter}")
    ROI_table = ad.AnnData(X=np.array(X, dtype=np.float32))
    ROI_table.obs_names = obs_names
    ROI_table.var_names = [
        "x_micrometer",
        "y_micrometer",
        "z_micrometer",
        "len_x_micrometer",
        "len_y_micrometer",
        "len_z_micrometer",
        "x_micrometer_original",
        "y_micrometer_original",
    ]
    return ROI_table


@validate_arguments
def import_ome_zarr(
    *,
    input_paths: Sequence[str],
    output_path: str,
    metadata: dict[str, Any],
    scope: Literal["plate", "well", "image"] = "plate",
    add_image_ROI_table: bool = True,
    add_grid_ROI_table: bool = True,
    grid_ROI_shape: Optional[tuple[int, int]] = None,
) -> dict[str, Any]:
    """
    Import an OME-Zarr

    Args:
        input_paths: TBD
            (standard argument for Fractal tasks, managed by Fractal server).
        output_path: TBD
            (standard argument for Fractal tasks, managed by Fractal server).
        component: TBD
            (standard argument for Fractal tasks, managed by Fractal server).
        metadata: TBD
            (standard argument for Fractal tasks, managed by Fractal server).
        scope: TBD (FIXME: rename it)
        add_image_ROI_table:
        add_grid_ROI_table:
        grid_ROI_shape:
    """

    # Preliminary checks
    if len(input_paths) > 1:
        raise NotImplementedError

    zarr_path = input_paths[0]
    logger.info(f"{zarr_path=}")

    zarr_path_name = Path(zarr_path).name

    zarrurls: dict = dict(plate=[], well=[], image=[])
    if scope != "plate":
        raise NotImplementedError

    plate_group = zarr.open_group(zarr_path, mode="r")
    zarrurls["plate"].append(Path(zarr_path).name)
    well_list = plate_group.attrs["plate"]["wells"]
    for well in well_list:
        well_subpath = well["path"]

        well_group = zarr.open_group(zarr_path, path=well_subpath, mode="r")
        zarrurls["well"].append(f"{zarr_path_name}/{well_subpath}")
        image_list = well_group.attrs["well"]["images"]
        for ind_image, image in enumerate(image_list):
            image_subpath = image["path"]
            zarrurls["image"].append(
                f"{zarr_path_name}/{well_subpath}/{image_subpath}"
            )

            # open_group docs: "`r+` means read/write (must exist)"
            image_group = zarr.open_group(
                zarr_path, path=f"{well_subpath}/{image_subpath}", mode="r+"
            )
            # debug(image_group.attrs.asdict())
            image_meta = NgffImageMeta(**image_group.attrs.asdict())
            pixels_ZYX = image_meta.get_pixel_sizes_zyx(level=0)

            if ind_image == 0:
                dataset_subpath = image_meta.datasets[0].path
                array = da.from_zarr(
                    (
                        f"{zarr_path}/{well_subpath}/"
                        f"{image_subpath}/{dataset_subpath}"
                    )
                )
                image_ROI_table = _get_image_ROI_table(array.shape, pixels_ZYX)
                grid_ROI_table = _get_grid_ROI_table(
                    array.shape,
                    pixels_ZYX,
                    grid_ROI_shape,
                )
                write_table(
                    image_group,
                    "image_ROI_table",
                    image_ROI_table,
                    overwrite=True,  # FIXME: what should we add here?
                    logger=logger,
                )
                write_table(
                    image_group,
                    "grid_ROI_table",
                    grid_ROI_table,
                    overwrite=True,  # FIXME: what should we add here?
                    logger=logger,
                )

    clean_zarrurls = {k: v for k, v in zarrurls.items() if v}

    return clean_zarrurls


if __name__ == "__main__":

    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=import_ome_zarr,
        logger_name=logger.name,
    )

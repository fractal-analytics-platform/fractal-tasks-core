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
Task to import an existing OME-Zarr.
"""
import logging
from pathlib import Path
from typing import Any
from typing import Sequence

import dask.array as da
import zarr
from pydantic.decorator import validate_arguments

from fractal_tasks_core.lib_ngff import detect_ome_ngff_type
from fractal_tasks_core.lib_ngff import NgffImageMeta
from fractal_tasks_core.lib_regions_of_interest import get_image_grid_ROIs
from fractal_tasks_core.lib_regions_of_interest import get_single_image_ROI
from fractal_tasks_core.lib_write import write_table

logger = logging.getLogger(__name__)


def _add_ROI_tables(
    image_path: str,
    add_image_ROI_table: bool,
    add_grid_ROI_table: bool,
    grid_YX_shape: tuple[int, int],
) -> None:
    """
    Generate ROI table for an OME-NGFF image

    Args:
        image_path: Absolute path to the image Zarr group.
        add_image_ROI_table: Whether to add a `image_ROI_table` table
            (argument propagated from `import_ome_zarr`).
        add_grid_ROI_table: Whether to add a `grid_ROI_table` table (argument
            propagated from `import_ome_zarr`).
        grid_YX_shape: YX shape of the ROI grid.
    """

    # Note from zar docs: `r+` means read/write (must exist)
    image_group = zarr.open_group(image_path, mode="r+")
    image_meta = NgffImageMeta(**image_group.attrs.asdict())

    if not (add_image_ROI_table or add_grid_ROI_table):
        return

    pixels_ZYX = image_meta.get_pixel_sizes_zyx(level=0)

    # Read zarr array
    dataset_subpath = image_meta.datasets[0].path
    array = da.from_zarr(f"{image_path}/{dataset_subpath}")

    # Prepare image_ROI_table and write it into the zarr group
    if add_image_ROI_table:
        image_ROI_table = get_single_image_ROI(array.shape, pixels_ZYX)
        write_table(
            image_group,
            "image_ROI_table",
            image_ROI_table,
            overwrite=False,
            logger=logger,
        )

    # Prepare grid_ROI_table and write it into the zarr group
    if add_grid_ROI_table:
        grid_ROI_table = get_image_grid_ROIs(
            array.shape,
            pixels_ZYX,
            grid_YX_shape,
        )
        write_table(
            image_group,
            "grid_ROI_table",
            grid_ROI_table,
            overwrite=False,
            logger=logger,
        )


@validate_arguments
def import_ome_zarr(
    *,
    input_paths: Sequence[str],
    output_path: str,
    metadata: dict[str, Any],
    zarr_name: str,
    add_image_ROI_table: bool = True,
    add_grid_ROI_table: bool = True,
    grid_y_shape: int = 2,
    grid_x_shape: int = 2,
) -> dict[str, Any]:
    """
    Import an OME-Zarr into Fractal.

    The current version of this task:

    1. Creates the appropriate components-related metadata, needed for
       processing an existing OME-Zarr through Fractal.
    2. Optionally adds new ROI tables to the existing OME-Zarr.

    Args:
        input_paths: A length-one list with the parent folder of the OME-Zarr
            to be imported; e.g. `input_paths=["/somewhere"]`, if the OME-Zarr
            path is `/somewhere/array.zarr`.
            (standard argument for Fractal tasks, managed by Fractal server).
        output_path: Not used in this task.
            (standard argument for Fractal tasks, managed by Fractal server).
        metadata: Not used in this task.
            (standard argument for Fractal tasks, managed by Fractal server).
        zarr_name: The OME-Zarr name, without its parent folder; e.g.
            `zarr_name="array.zarr"`, if the OME-Zarr path is
            `/somewhere/array.zarr`.
        add_image_ROI_table: Whether to add a `image_ROI_table` table to each
            image, with a single ROI covering the whole image.
        add_grid_ROI_table: Whether to add a `grid_ROI_table` table to each
            image, with the image split into a rectangular grid of ROIs.
        grid_y_shape: Y shape of the ROI grid in `grid_ROI_shape`.
        grid_x_shape: X shape of the ROI grid in `grid_ROI_shape`.
    """

    # Preliminary checks
    if len(input_paths) > 1:
        raise NotImplementedError

    zarr_path = str(Path(input_paths[0]) / zarr_name)
    logger.info(f"Zarr path: {zarr_path}")

    zarrurls: dict = dict(plate=[], well=[], image=[])

    root_group = zarr.open_group(zarr_path, mode="r")
    ngff_type = detect_ome_ngff_type(root_group)
    grid_YX_shape = (grid_y_shape, grid_x_shape)

    if ngff_type == "plate":
        zarrurls["plate"].append(zarr_name)
        for well in root_group.attrs["plate"]["wells"]:
            well_path = well["path"]
            zarrurls["well"].append(f"{zarr_name}/{well_path}")

            well_group = zarr.open_group(zarr_path, path=well_path, mode="r")
            for image in well_group.attrs["well"]["images"]:
                image_path = image["path"]
                zarrurls["image"].append(
                    f"{zarr_name}/{well_path}/{image_path}"
                )
                _add_ROI_tables(
                    f"{zarr_path}/{well_path}/{image_path}",
                    add_image_ROI_table,
                    add_grid_ROI_table,
                    grid_YX_shape,
                )
    elif ngff_type == "well":
        zarrurls["well"].append(zarr_name)
        logger.warning(
            "Only OME-Zarr for plates are fully supported in Fractal; "
            "e.g. the current one ({ngff_type=}) cannot be "
            "processed via the `maximum_intensity_projection` task."
        )
        for image in root_group.attrs["well"]["images"]:
            image_path = image["path"]
            zarrurls["image"].append(f"{zarr_name}/{image_path}")
            _add_ROI_tables(
                f"{zarr_path}/{image_path}",
                add_image_ROI_table,
                add_grid_ROI_table,
                grid_YX_shape,
            )
    elif ngff_type == "image":
        zarrurls["image"].append(zarr_name)
        logger.warning(
            "Only OME-Zarr for plates are fully supported in Fractal; "
            "e.g. the current one ({ngff_type=}) cannot be "
            "processed via the `maximum_intensity_projection` task."
        )
        _add_ROI_tables(
            zarr_path,
            add_image_ROI_table,
            add_grid_ROI_table,
            grid_YX_shape,
        )

    # Remove zarrurls keys pointing to empty lists
    clean_zarrurls = {
        key: value for key, value in zarrurls.items() if len(value) > 0
    }

    return clean_zarrurls


if __name__ == "__main__":

    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=import_ome_zarr,
        logger_name=logger.name,
    )

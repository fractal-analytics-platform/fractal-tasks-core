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
from typing import Any

import anndata as ad
import dask.array as da
import zarr
from pydantic import validate_call
from zarr.errors import ContainsArrayError

from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.pyramids import build_pyramid
from fractal_tasks_core.roi import (
    convert_ROIs_from_3D_to_2D,
)
from fractal_tasks_core.tables import write_table
from fractal_tasks_core.tables.v1 import get_tables_list_v1
from fractal_tasks_core.tasks.io_models import InitArgsMIP
from fractal_tasks_core.zarr_utils import OverwriteNotAllowedError


logger = logging.getLogger(__name__)


@validate_call
def maximum_intensity_projection(
    *,
    # Fractal parameters
    zarr_url: str,
    init_args: InitArgsMIP,
    # Advanced parameters
    overwrite: bool = False,
) -> dict[str, Any]:
    """
    Perform maximum-intensity projection along Z axis.

    Note: this task stores the output in a new zarr file.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        init_args: Intialization arguments provided by
            `create_cellvoyager_ome_zarr_init`.
        overwrite: If `True`, overwrite the task output.
    """
    logger.info(f"{init_args.origin_url=}")
    logger.info(f"{zarr_url=}")

    # Read image metadata
    ngff_image = load_NgffImageMeta(init_args.origin_url)
    # Currently not using the validation models due to wavelength_id issue
    # See #681 for discussion
    # new_attrs = ngff_image.model_dump(exclude_none=True)
    # Current way to get the necessary metadata for MIP
    group = zarr.open_group(init_args.origin_url, mode="r")
    new_attrs = group.attrs.asdict()

    # Create the zarr image with correct
    new_image_group = zarr.group(zarr_url)
    new_image_group.attrs.put(new_attrs)

    # Load 0-th level
    data_czyx = da.from_zarr(init_args.origin_url + "/0")
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
            f"{zarr_url}/0",
            overwrite=overwrite,
            dimension_separator="/",
            write_empty_chunks=False,
        )
    except ContainsArrayError as e:
        error_msg = (
            f"Cannot write array to zarr group at '{zarr_url}/0', "
            f"with {overwrite=} (original error: {str(e)}).\n"
            "Hint: try setting overwrite=True."
        )
        logger.error(error_msg)
        raise OverwriteNotAllowedError(error_msg)

    # Starting from on-disk highest-resolution data, build and write to disk a
    # pyramid of coarser levels
    build_pyramid(
        zarrurl=zarr_url,
        overwrite=overwrite,
        num_levels=ngff_image.num_levels,
        coarsening_xy=ngff_image.coarsening_xy,
        chunksize=(1, 1, chunksize_y, chunksize_x),
    )

    # Copy over any tables from the original zarr
    # Generate the list of tables:
    tables = get_tables_list_v1(init_args.origin_url)
    roi_tables = get_tables_list_v1(init_args.origin_url, table_type="ROIs")
    non_roi_tables = [table for table in tables if table not in roi_tables]

    for table in roi_tables:
        logger.info(
            f"Reading {table} from "
            f"{init_args.origin_url=}, convert it to 2D, and "
            "write it back to the new zarr file."
        )
        new_ROI_table = ad.read_zarr(f"{init_args.origin_url}/tables/{table}")
        old_ROI_table_attrs = zarr.open_group(
            f"{init_args.origin_url}/tables/{table}"
        ).attrs.asdict()

        # Convert 3D ROIs to 2D
        pxl_sizes_zyx = ngff_image.get_pixel_sizes_zyx(level=0)
        new_ROI_table = convert_ROIs_from_3D_to_2D(
            new_ROI_table, pixel_size_z=pxl_sizes_zyx[0]
        )
        # Write new table
        write_table(
            new_image_group,
            table,
            new_ROI_table,
            table_attrs=old_ROI_table_attrs,
            overwrite=overwrite,
        )

    for table in non_roi_tables:
        logger.info(
            f"Reading {table} from "
            f"{init_args.origin_url=}, and "
            "write it back to the new zarr file."
        )
        new_non_ROI_table = ad.read_zarr(
            f"{init_args.origin_url}/tables/{table}"
        )
        old_non_ROI_table_attrs = zarr.open_group(
            f"{init_args.origin_url}/tables/{table}"
        ).attrs.asdict()

        # Write new table
        write_table(
            new_image_group,
            table,
            new_non_ROI_table,
            table_attrs=old_non_ROI_table_attrs,
            overwrite=overwrite,
        )

    # Generate image_list_updates
    image_list_update_dict = dict(
        image_list_updates=[
            dict(
                zarr_url=zarr_url,
                origin=init_args.origin_url,
                types=dict(is_3D=False),
            )
        ]
    )
    return image_list_update_dict


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=maximum_intensity_projection,
        logger_name=logger.name,
    )

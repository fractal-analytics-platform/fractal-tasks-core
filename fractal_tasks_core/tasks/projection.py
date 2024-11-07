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

import dask.array as da
from ngio import NgffImage
from pydantic import validate_call

from fractal_tasks_core.tasks.io_models import InitArgsMIP
from fractal_tasks_core.tasks.projection_utils import DaskProjectionMethod

logger = logging.getLogger(__name__)


@validate_call
def projection(
    *,
    # Fractal parameters
    zarr_url: str,
    init_args: InitArgsMIP,
) -> dict[str, Any]:
    """
    Perform intensity projection along Z axis with a chosen method.

    Note: this task stores the output in a new zarr file.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        init_args: Intialization arguments provided by
            `create_cellvoyager_ome_zarr_init`.
    """
    method = DaskProjectionMethod(init_args.method)
    logger.info(f"{init_args.origin_url=}")
    logger.info(f"{zarr_url=}")
    logger.info(f"{method=}")

    # Read image metadata
    original_ngff_image = NgffImage(init_args.origin_url)
    orginal_image = original_ngff_image.get_image()

    if orginal_image.is_2d or orginal_image.is_2d_time_series:
        raise ValueError(
            "The input image is 2D, "
            "projection is only supported for 3D images."
        )

    on_disk_shape = orginal_image.on_disk_shape
    logger.info(f"Original shape: {on_disk_shape=}")

    on_disk_z_index = orginal_image.find_axis("z")

    new_on_disk_shape = list(on_disk_shape)
    new_on_disk_shape[on_disk_z_index] = 1

    pixel_size = orginal_image.pixel_size
    pixel_size.z = 1.0
    logger.info(f"New shape: {new_on_disk_shape=}")

    new_ngff_image = original_ngff_image.derive_new_image(
        store=zarr_url,
        name="MIP",
        on_disk_shape=new_on_disk_shape,
        pixel_sizes=pixel_size,
        overwrite=init_args.overwrite,
    )
    new_image = new_ngff_image.get_image()

    # Process the image
    z_axis_index = orginal_image.dataset.axes_names.index("z")
    source_dask = orginal_image.get_array(
        mode="dask", preserve_dimensions=True
    )

    dest_dask = method.apply(dask_array=source_dask, axis=z_axis_index)
    dest_dask = da.expand_dims(dest_dask, axis=z_axis_index)
    new_image.set_array(dest_dask)
    new_image.consolidate()
    # Ends

    # Copy over the tables
    for roi_table in original_ngff_image.table.list(table_type="roi_table"):
        table = original_ngff_image.table.get_table(roi_table)
        mip_table = new_ngff_image.table.new(
            roi_table, table_type="roi_table", overwrite=True
        )
        roi_list = []
        for roi in table.rois:
            roi.z_length = roi.z + 1
            roi_list.append(roi)

        mip_table.set_rois(roi_list, overwrite=True)
        mip_table.consolidate()

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
        task_function=projection,
        logger_name=logger.name,
    )

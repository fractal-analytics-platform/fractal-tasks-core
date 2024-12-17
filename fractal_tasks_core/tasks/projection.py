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
from __future__ import annotations

import logging
from typing import Any

import dask.array as da
from ngio import NgffImage
from ngio.core import Image
from pydantic import validate_call

from fractal_tasks_core.tasks.io_models import InitArgsMIP
from fractal_tasks_core.tasks.projection_utils import DaskProjectionMethod

logger = logging.getLogger(__name__)


def _compute_new_shape(source_image: Image) -> tuple[int]:
    """Compute the new shape of the image after the projection.

    The new shape is the same as the original one,
    except for the z-axis, which is set to 1.
    """
    on_disk_shape = source_image.on_disk_shape
    logger.info(f"Source {on_disk_shape=}")

    on_disk_z_index = source_image.dataset.on_disk_axes_names.index("z")

    dest_on_disk_shape = list(on_disk_shape)
    dest_on_disk_shape[on_disk_z_index] = 1
    logger.info(f"Destination {dest_on_disk_shape=}")
    return tuple(dest_on_disk_shape)


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

    # Compute the new shape and pixel size
    dest_on_disk_shape = _compute_new_shape(orginal_image)

    dest_pixel_size = orginal_image.pixel_size
    dest_pixel_size.z = 1.0
    logger.info(f"New shape: {dest_on_disk_shape=}")

    # Create the new empty image
    new_ngff_image = original_ngff_image.derive_new_image(
        store=zarr_url,
        name="MIP",
        on_disk_shape=dest_on_disk_shape,
        pixel_sizes=dest_pixel_size,
        overwrite=init_args.overwrite,
        copy_labels=False,
        copy_tables=True,
    )
    logger.info(f"New Projection image created - {new_ngff_image=}")
    new_image = new_ngff_image.get_image()

    # Process the image
    z_axis_index = orginal_image.find_axis("z")
    source_dask = orginal_image.get_array(
        mode="dask", preserve_dimensions=True
    )

    dest_dask = method.apply(dask_array=source_dask, axis=z_axis_index)
    dest_dask = da.expand_dims(dest_dask, axis=z_axis_index)
    new_image.set_array(dest_dask)
    new_image.consolidate()
    # Ends

    # Copy over the tables
    for roi_table_name in new_ngff_image.tables.list(table_type="roi_table"):
        table = new_ngff_image.tables.get_table(roi_table_name)

        roi_list = []
        for roi in table.rois:
            roi.z = 0.0
            roi.z_length = 1.0
            roi_list.append(roi)

        table.set_rois(roi_list, overwrite=True)
        table.consolidate()
        logger.info(f"Table {roi_table_name} Projection done")

    # Generate image_list_updates
    image_list_update_dict = dict(
        image_list_updates=[
            dict(
                zarr_url=zarr_url,
                origin=init_args.origin_url,
                attributes=dict(plate=init_args.new_plate_name),
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

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
from typing import Literal

import dask.array as da
import numpy as np
import skimage.transform
from ngio import create_empty_ome_zarr
from ngio import Image
from ngio import open_ome_zarr_container
from ngio import PixelSize
from ngio.common import Roi
from ngio.ome_zarr_meta import get_image_meta_handler
from ngio.ome_zarr_meta import NgioImageMeta
from pydantic import validate_call

from fractal_tasks_core.tasks.io_models import InitArgsMIP
from fractal_tasks_core.tasks.projection_utils import DaskProjectionMethod

logger = logging.getLogger(__name__)


def _project_roi(roi: Roi, projection_axis: Literal["z", "y", "x"]) -> Roi:
    """Project the ROI along the given axis,
    assumes rotation to YX plane if needed."""
    if projection_axis == "x":
        roi.z, roi.y, roi.x = 0.0, roi.z, roi.y
        roi.z_length, roi.y_length, roi.x_length = (
            1.0,
            roi.z_length,
            roi.y_length,
        )
    elif projection_axis == "y":
        roi.z, roi.y, roi.x = 0.0, roi.x, roi.z
        roi.z_length, roi.y_length, roi.x_length = (
            1.0,
            roi.x_length,
            roi.z_length,
        )
    elif projection_axis == "z":
        roi.z = 0.0
        roi.z_length = 1.0
    return roi


def _rotate_axes_values(
    source_image: Image,
    input_axes_values: tuple,
    projection_axis: Literal["z", "y", "x"],
):
    """Rotate axes-related values (e.g. shape) to have projection axis always
    on z-axis index by coordinates rotation"""
    z_axis_index = source_image.axes_handler.get_index("z")
    y_axis_index = source_image.axes_handler.get_index("y")
    x_axis_index = source_image.axes_handler.get_index("x")

    output_axes_values = list(input_axes_values)
    if projection_axis == "y":
        output_axes_values[z_axis_index] = input_axes_values[y_axis_index]
        output_axes_values[y_axis_index] = input_axes_values[x_axis_index]
        output_axes_values[x_axis_index] = input_axes_values[z_axis_index]
    elif projection_axis == "x":
        output_axes_values[z_axis_index] = input_axes_values[x_axis_index]
        output_axes_values[y_axis_index] = input_axes_values[z_axis_index]
        output_axes_values[x_axis_index] = input_axes_values[y_axis_index]
    return tuple(output_axes_values)


def _compute_new_shape(
    source_image: Image,
    projection_axis: Literal["z", "y", "x"],
    z_upscale_factor: float,
) -> tuple[
    tuple[int, ...], tuple[int, ...], tuple[int, ...], int, tuple[int, ...]
]:
    """Compute the new shape of the image after the projection.

    The new shape is the same as the original one,
    except for the projection-axis, which is set to 1.
    In case of projection along "x" or "y", the axes are rotated
    so that the non-singular spatial dimensions are always YX.

    returns:
        - new shape of the image
        - index of the projection-axis in the original image
        - axes rotation to apply after projection (None if no rotation needed)
    """
    on_disk_shape = source_image.shape
    on_disk_chunks = source_image.chunks
    on_disk_scaling = source_image.meta.scaling_factor()
    logger.info(f"Source {on_disk_shape=}")

    z_axis_index: int = source_image.axes_handler.get_index("z")
    projection_index = source_image.axes_handler.get_index(projection_axis)
    if projection_index is None:
        raise ValueError(
            f"The input image does not contain a {projection_axis}-axis, "
            f"projection is only supported for 3D images with "
            f"a {projection_axis}-axis."
        )

    dest_shape = list(on_disk_shape)
    dest_shape[projection_index] = 1

    dest_chunks = list(on_disk_chunks)
    dest_chunks[projection_index] = 1

    dest_scaling = list(on_disk_scaling)
    dest_scaling[projection_index] = 1.0

    # Make a rotation if needed
    axes_rotation = list(range(len(dest_shape)))
    if projection_axis != "z":
        dest_shape[z_axis_index] = round(
            on_disk_shape[z_axis_index] * z_upscale_factor
        )

        axes_rotation = _rotate_axes_values(
            source_image,
            axes_rotation,
            projection_axis,
        )

        dest_shape = _rotate_axes_values(
            source_image,
            dest_shape,
            projection_axis,
        )

        dest_chunks = _rotate_axes_values(
            source_image,
            dest_chunks,
            projection_axis,
        )

        dest_scaling = _rotate_axes_values(
            source_image,
            dest_scaling,
            projection_axis,
        )

    logger.info(f"Destination {dest_shape=}")
    return (
        tuple(dest_shape),
        tuple(dest_chunks),
        tuple(dest_scaling),
        z_axis_index,
        axes_rotation,
    )


def _compute_image_sharpness(image: np.array) -> np.float32:
    """Compute sharpness map based on mean gradient magnitude of an image.

    Assuming image has shape (y, x), we compute the gradient
    along y and x axes and then compute the gradient magnitude
    to obtain a sharpness value.

    Args:
        image (np.ndarray): The input image.
    Returns:
        np.ndarray: The sharpness value.
    """
    gradient: tuple[np.ndarray, np.ndarray] = np.gradient(image)
    gradient_norm: np.ndarray = np.sqrt(gradient[0] ** 2 + gradient[1] ** 2)
    return np.mean(gradient_norm)


def _project_block_with_autofocus(
    block: np.ndarray,
    z_indices: np.ndarray,
    z_mask: np.ndarray | None,
    projection_method: DaskProjectionMethod,
) -> np.ndarray:
    # TODO(PR): review docs
    """Project a Dask array block with optional autofocus.

    Assuming block has shape (t, c, z, y, x), we make a projection
    along the z axis using only a window of slices around the sharpest plane,
    determined via autofocus.

    Args:
        dask_array (da.Array): The input Dask array block.
        z_indices (np.ndarray): The z indices to use for projection.
        z_mask (np.ndarray | None): The mask to apply for mean/sum projection.

    Returns:
        da.Array: The projected Dask array block.
    """

    # Extract the window
    t_grid = np.arange(block.shape[0])[:, None, None]
    c_grid = np.arange(block.shape[1])[None, :, None]
    window = block[t_grid, c_grid, z_indices, :, :]

    # In case of mean/sum projection, we need to mask the clipped values
    if projection_method in (
        DaskProjectionMethod.MEANIP,
        DaskProjectionMethod.SUMIP,
    ):
        window = window * z_mask[..., None, None]

    return np.asarray(projection_method.apply(window, axis=2))


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
    advanced_parameters = init_args.advanced_parameters

    logger.info(f"{init_args.origin_url=}")
    logger.info(f"{zarr_url=}")
    logger.info(f"{method=}")
    logger.info(f"{advanced_parameters=}")

    # Read image metadata
    original_ome_zarr = open_ome_zarr_container(init_args.origin_url)
    original_image = original_ome_zarr.get_image()

    if original_image.is_2d or original_image.is_2d_time_series:
        raise ValueError(
            "The input image is 2D, "
            "projection is only supported for 3D images."
        )

    # Compute the new shape, new pixel size, scaling factors and axes rotation
    (
        dest_shape,
        dest_chunks,
        dest_scaling,
        z_axis_index,
        axes_rotation,
    ) = _compute_new_shape(
        original_image,
        advanced_parameters.projection_axis,
        advanced_parameters.z_upscale_factor,
    )

    logger.info(f"New shape: {dest_shape}")
    logger.info(f"New chunks: {dest_chunks}.")
    logger.info(f"New scaling factors: {dest_scaling}.")
    logger.info(f"Projection with {axes_rotation=}.")

    original_pixel_size = original_image.pixel_size
    if advanced_parameters.projection_axis == "x":
        new_pixel_size = PixelSize(
            t=original_pixel_size.t,
            z=1.0,
            y=original_pixel_size.z / advanced_parameters.z_upscale_factor,
            x=original_pixel_size.y,
            space_unit=original_pixel_size.space_unit,
            time_unit=original_pixel_size.time_unit,
        )
    elif advanced_parameters.projection_axis == "y":
        new_pixel_size = PixelSize(
            t=original_pixel_size.t,
            z=1.0,
            y=original_pixel_size.x,
            x=original_pixel_size.z / advanced_parameters.z_upscale_factor,
            space_unit=original_pixel_size.space_unit,
            time_unit=original_pixel_size.time_unit,
        )
    else:  # projection along z
        new_pixel_size = PixelSize(
            t=original_pixel_size.t,
            z=1.0,
            y=original_pixel_size.y,
            x=original_pixel_size.x,
            space_unit=original_pixel_size.space_unit,
            time_unit=original_pixel_size.time_unit,
        )

    x_axis_index = original_image.axes_handler.get_index("x")
    y_axis_index = original_image.axes_handler.get_index("y")
    z_axis_index = original_image.axes_handler.get_index("z")
    xy_scaling = (dest_scaling[y_axis_index], dest_scaling[x_axis_index])
    z_scaling = dest_scaling[z_axis_index]

    # Create the new empty image
    ome_zarr_mip = create_empty_ome_zarr(
        store=zarr_url,
        shape=dest_shape,
        xy_pixelsize=new_pixel_size.x,
        z_spacing=new_pixel_size.z,
        time_spacing=new_pixel_size.t,
        levels=original_ome_zarr.levels,
        xy_scaling_factor=xy_scaling,
        z_scaling_factor=z_scaling,
        space_unit=new_pixel_size.space_unit,
        time_unit=new_pixel_size.time_unit,
        axes_names=original_image.axes_handler.axes_names,
        name="MIP",
        chunks=dest_chunks,
        dtype=original_image.dtype,
        overwrite=init_args.overwrite,
    )

    logger.info(f"New Projection image created - {ome_zarr_mip=}")
    proj_image = ome_zarr_mip.get_image()

    # We need to edit pixel_size for rotated projections
    # because this cannot be set at creation
    if advanced_parameters.projection_axis != "z":
        image_meta_handler = get_image_meta_handler(
            group_handler=ome_zarr_mip._group_handler,
            version=original_image.meta.version,
        )

        new_meta = NgioImageMeta.default_init(
            levels=ome_zarr_mip.image_meta.levels,
            axes_names=original_image.axes_handler.axes_names,
            pixel_size=new_pixel_size,
            scaling_factors=dest_scaling,
            name=ome_zarr_mip.image_meta.name,
            version=original_image.meta.version,
        )

        image_meta_handler.write_meta(new_meta)

    # Copy channels metadata
    original_meta = original_image.meta.channels_meta
    ome_zarr_mip.set_channel_meta(
        labels=original_image.channel_labels,
        wavelength_id=original_image.wavelength_ids,
        percentiles=None,
        colors=[c.channel_visualisation.color for c in original_meta.channels],
        active=[
            c.channel_visualisation.active for c in original_meta.channels
        ],
    )

    # Copy tables from original image
    original_ome_zarr.tables_container._group_handler.copy_handler(
        ome_zarr_mip.tables_container._group_handler
    )

    # Process the image
    dask_array = original_image.get_as_dask()

    # Rotate axes to have projection axis always on z-axis index
    if advanced_parameters.projection_axis != "z":
        dask_array = da.transpose(dask_array, axes=axes_rotation)

    # Apply the projection
    if (
        advanced_parameters.autofocus_radius is None
        or advanced_parameters.autofocus_radius
        >= dask_array.shape[z_axis_index]
    ):  # Simple projection without autofocus
        dask_array = method.apply(dask_array=dask_array, axis=z_axis_index)
    else:  # Autofocus projection needs some tricks to do efficiently
        # Ensure array has 5 dimensions (t, c, z, y, x) for autofocus
        # Possibly could be optimized further by smarter indexing
        if original_image.axes_handler.axes_names[-3:] != ("z", "y", "x"):
            raise ValueError(
                "Autofocus projection is only supported for images "
                "with canonical axes ordering: ...zyx"
            )
        original_ndim = dask_array.ndim
        while dask_array.ndim < 5:
            dask_array = da.expand_dims(dask_array, axis=0)

        sharpness_map: da.Array = da.apply_gufunc(
            _compute_image_sharpness,
            "(y, x) -> ()",
            dask_array.rechunk({3: -1, 4: -1}),  # rechunk to full (y,x) image
            vectorize=True,
            output_dtypes=float,
        )

        # Generate the window with focused planes indices
        radius: int = advanced_parameters.autofocus_radius
        center_indices: da.Array = sharpness_map.argmax(axis=-1)
        z_indices: da.Array = center_indices[..., None] + np.arange(
            -radius, radius + 1
        )

        # Handle Boundary Conditions (Clipping)
        # when sharpest plane is close to boundaries we need to clip indices
        # having repeating values at the edges is fine for dask computations.
        # This works for min and max projection methods, but not for mean
        # and sum - for those we create a mask to ignore the clipped values.
        z_max: int = dask_array.shape[2] - 1
        z_mask: da.Array | None = None
        if method in (DaskProjectionMethod.MEANIP, DaskProjectionMethod.SUMIP):
            z_mask = np.ones_like(z_indices, dtype=np.bool)
            z_mask[z_indices < 0] = 0
            z_mask[z_indices > z_max] = 0
        z_indices = np.clip(z_indices, 0, z_max)

        # Do projection with only z-axis not chunked
        dask_array = da.map_blocks(
            _project_block_with_autofocus,
            dask_array,
            z_indices[..., None, None],
            z_mask[..., None, None] if z_mask is not None else None,
            method,
            drop_axis=2,  # drop z axis after projection
            dtype=dask_array.dtype,
            chunks=(
                dask_array.chunks[0],
                dask_array.chunks[1],
                dask_array.chunks[3],
                dask_array.chunks[4],
            ),
        )

        # Remove any dimensions that were added
        while dask_array.ndim > original_ndim - 1:
            dask_array = da.squeeze(dask_array, axis=0)

    # Add dropped z-axis back
    dask_array = da.expand_dims(dask_array, axis=z_axis_index)

    # Apply z-upscaling if needed
    if (
        advanced_parameters.projection_axis != "z"
        and advanced_parameters.z_upscale_factor != 1.0
    ):
        logger.info("Applying z-upscaling after projection.")
        logger.info(f"Image shape before upscaling: {dask_array.shape}")

        # prepare scaler on the new Z-axis position
        scaler: list[float] = [1.0] * dask_array.ndim
        new_z_axis_index: int = axes_rotation.index(z_axis_index)
        scaler[new_z_axis_index] = advanced_parameters.z_upscale_factor

        dask_array = skimage.transform.rescale(
            dask_array,
            scale=tuple(scaler),
            order=advanced_parameters.z_upscale_interpolation_order,
        )
        logger.info(f"Image shape after upscaling: {dask_array.shape}")

    # Save the projected image
    proj_image.set_array(dask_array)
    proj_image.consolidate()
    # Ends

    # Edit the roi tables
    for roi_table_name in ome_zarr_mip.list_roi_tables():
        table = ome_zarr_mip.get_generic_roi_table(roi_table_name)

        for roi in table.rois():
            roi = _project_roi(roi, advanced_parameters.projection_axis)
            table.add(roi, overwrite=True)

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
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=projection,
        logger_name=logger.name,
    )

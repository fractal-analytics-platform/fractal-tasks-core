import logging
from enum import Enum
from typing import Any

import dask.array as da
import numpy as np
from ngio import Image, open_ome_zarr_container
from pydantic import BaseModel

logger = logging.getLogger("projection_utils")


def safe_sum(dask_array: da.Array, axis: int = 0) -> da.Array:
    """
    Perform a safe sum on a Dask array to avoid overflow, by clipping the
    result of da.sum & casting it to its original dtype.

    Dask.array already correctly handles promotion to uin32 or uint64 when
    necessary internally, but we want to ensure we clip the result.

    Args:
        dask_array (dask.array.Array): The input Dask array.
        axis (int, optional): The axis along which to sum the array.
            Defaults to 0.

    Returns:
        dask.array.Array: The result of the sum, safely clipped and cast
            back to the original dtype.
    """
    # Determine the original dtype
    original_dtype = dask_array.dtype
    if not np.issubdtype(original_dtype, np.integer):
        raise ValueError(
            f"safe_sum only supports integer dtypes, got {original_dtype}. "
            "Use a different projection method for float arrays."
        )
    max_value = np.iinfo(original_dtype).max

    # Perform the sum
    result = da.sum(dask_array, axis=axis)

    # Clip the values to the maximum possible value for the original dtype
    result = da.clip(result, 0, max_value)

    # Cast back to the original dtype
    result = result.astype(original_dtype)

    return result


def mean_wrapper(dask_array: da.Array, axis: int = 0) -> da.Array:
    """
    Perform a da.mean on the dask_array & cast it to its original dtype.

    Without casting, the result can change dtype to e.g. float64

    Args:
        dask_array (dask.array.Array): The input Dask array.
        axis (int, optional): The axis along which to mean the array.
            Defaults to 0.

    Returns:
        dask.array.Array: The result of the mean, cast back to the original
            dtype.
    """
    # Determine the original dtype
    original_dtype = dask_array.dtype

    # Perform the mean
    result = da.mean(dask_array, axis=axis)

    # Cast back to the original dtype
    result = result.astype(original_dtype)

    return result


def max_wrapper(dask_array: da.Array, axis: int = 0) -> da.Array:
    """
    Perform a da.max on the dask_array

    Args:
        dask_array (dask.array.Array): The input Dask array.
        axis (int, optional): The axis along which to max the array.
            Defaults to 0.

    Returns:
        dask.array.Array: The result of the max
    """
    return dask_array.max(axis=axis)


def min_wrapper(dask_array: da.Array, axis: int = 0) -> da.Array:
    """
    Perform a da.min on the dask_array

    Args:
        dask_array (dask.array.Array): The input Dask array.
        axis (int, optional): The axis along which to min the array.
            Defaults to 0.

    Returns:
        dask.array.Array: The result of the min
    """
    return dask_array.min(axis=axis)


class DaskProjectionMethod(Enum):
    """
    Registration method selection

    Choose which method to use for intensity projection along the Z axis.

    Attributes:
        MIP: Maximum intensity projection
        MINIP: Minimum intensity projection
        MEANIP: Mean intensity projection
        SUMIP: Sum intensity projection
    """

    MIP = "mip"
    MINIP = "minip"
    MEANIP = "meanip"
    SUMIP = "sumip"

    def apply(self, dask_array: da.Array, axis: int = 0) -> da.Array:
        """
        Apply the selected projection method to the given Dask array.

        Args:
            dask_array (dask.array.Array): The Dask array to project.
            axis (int): The axis along which to apply the projection.

        Returns:
            dask.array.Array: The resulting Dask array after applying the
                projection.
        """
        # Map the Enum values to the actual Dask array methods
        method_map = {
            DaskProjectionMethod.MIP: max_wrapper,
            DaskProjectionMethod.MINIP: min_wrapper,
            DaskProjectionMethod.MEANIP: mean_wrapper,
            DaskProjectionMethod.SUMIP: safe_sum,
        }
        # Call the appropriate method, passing in the dask_array explicitly
        return method_map[self](dask_array, axis=axis)


def _compute_new_shape(source_image: Image) -> tuple[tuple[int, ...], int]:
    """Compute the new shape of the image after the projection.

    The new shape is the same as the original one,
    except for the z-axis, which is set to 1.

    returns:
        - new shape of the image
        - index of the z-axis in the original image
    """
    on_disk_shape = source_image.shape
    logger.info(f"Source {on_disk_shape=}")

    on_disk_z_index = source_image.axes_handler.get_index("z")
    if on_disk_z_index is None:
        raise ValueError(
            "The input image does not contain a z-axis, "
            "projection is only supported for 3D images with a z-axis."
        )

    dest_on_disk_shape = list(on_disk_shape)
    dest_on_disk_shape[on_disk_z_index] = 1
    logger.info(f"Destination {dest_on_disk_shape=}")
    return tuple(dest_on_disk_shape), on_disk_z_index


def projection_core(
    *,
    input_zarr_url: str,
    output_zarr_url: str,
    method: DaskProjectionMethod = DaskProjectionMethod.MIP,
    overwrite: bool = False,
    attributes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Perform intensity projection along Z axis with a chosen method.

    Note: this task stores the output in a new zarr file.

    Args:
        input_zarr_url: Path or url to the individual OME-Zarr image to be processed.
        output_zarr_url: Path or url to the output OME-Zarr image.
        method: Projection method to be used. See `DaskProjectionMethod`
        overwrite: If `True`, overwrite the task output.
        attributes: Additional attributes to be added to the output image.
    """
    logger.info(f"{input_zarr_url=}")
    logger.info(f"{output_zarr_url=}")
    logger.info(f"{method=}")

    # Read image metadata
    original_ome_zarr = open_ome_zarr_container(input_zarr_url)
    orginal_image = original_ome_zarr.get_image()

    if orginal_image.is_2d or orginal_image.is_2d_time_series:
        raise ValueError(
            "The input image is 2D, projection is only supported for 3D images."
        )

    # Compute the new shape and pixel size
    dest_on_disk_shape, z_axis_index = _compute_new_shape(orginal_image)
    logger.info(f"New shape: {dest_on_disk_shape=}")

    # Create the new empty image
    ome_zarr_mip = original_ome_zarr.derive_image(
        store=output_zarr_url,
        name=method.value.upper(),
        shape=dest_on_disk_shape,
        pixelsize=orginal_image.pixel_size.yx,
        z_spacing=1.0,
        time_spacing=orginal_image.pixel_size.t,
        overwrite=overwrite,
        copy_labels=False,
        copy_tables=True,
    )
    logger.info(f"New Projection image created - {ome_zarr_mip=}")
    proj_image = ome_zarr_mip.get_image()

    # Process the image
    source_dask = orginal_image.get_as_dask()
    dest_dask = method.apply(dask_array=source_dask, axis=z_axis_index)
    dest_dask = da.expand_dims(dest_dask, axis=z_axis_index)
    proj_image.set_array(dest_dask)
    proj_image.consolidate()
    # Ends

    # Edit the roi tables
    for roi_table_name in ome_zarr_mip.list_roi_tables():
        table = ome_zarr_mip.get_generic_roi_table(roi_table_name)

        for roi in table.rois():
            old_z_slice = roi.get("z")
            if old_z_slice is not None:
                roi = roi.update_slice("z", (0, 1))
            table.add(roi, overwrite=True)

        table.consolidate()
        logger.info(f"Table {roi_table_name} Projection done")

    # Generate image_list_updates
    attributes = attributes or {}
    image_list_update_dict = dict(
        image_list_updates=[
            dict(
                zarr_url=output_zarr_url,
                origin=input_zarr_url,
                attributes=attributes,
                types=dict(is_3D=False),
            )
        ]
    )
    return image_list_update_dict


class InitArgsMIP(BaseModel):
    """
    Init Args for MIP task.

    Attributes:
        origin_url: Path to the zarr_url with the 3D data
        method: Projection method to be used. See `DaskProjectionMethod`
        overwrite: If `True`, overwrite the task output.
        new_plate_name: Name of the new OME-Zarr HCS plate
    """

    origin_url: str
    method: str
    overwrite: bool
    new_plate_name: str

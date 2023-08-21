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
Function to increase the shape of an array by replicating it.
"""
import logging
import warnings
from typing import Optional
from typing import Sequence

import numpy as np


def upscale_array(
    *,
    array: np.ndarray,
    target_shape: tuple[int, ...],
    axis: Optional[Sequence[int]] = None,
    pad_with_zeros: bool = False,
    warn_if_inhomogeneous: bool = False,
) -> np.ndarray:
    """
    Upscale an array along a given list of axis (through repeated application
    of `np.repeat`), to match a target shape.

    Args:
        array: The array to be upscaled.
        target_shape: The shape of the rescaled array.
        axis: The axis along which to upscale the array (if `None`, then all
            axis are used).
        pad_with_zeros: If `True`, pad the upscaled array with zeros to match
            `target_shape`.
        warn_if_inhomogeneous: If `True`, raise a warning when the conversion
            factors are not identical across all dimensions.

    Returns:
        The upscaled array, with shape `target_shape`.
    """

    # Default behavior: use all axis
    if axis is None:
        axis = list(range(len(target_shape)))

    array_shape = array.shape
    info = (
        f"Trying to upscale from {array_shape=} to {target_shape=}, "
        f"acting on {axis=}."
    )

    if len(array_shape) != len(target_shape):
        raise ValueError(f"{info} Dimensions-number mismatch.")
    if axis == []:
        raise ValueError(f"{info} Empty axis list")
    if min(axis) < 0:
        raise ValueError(f"{info} Negative axis specification not allowed.")

    # Check that upscale is doable
    for ind, dim in enumerate(array_shape):
        # Check that array is not larger than target (downscaling)
        if dim > target_shape[ind]:
            raise ValueError(
                f"{info} {ind}-th array dimension is larger than target."
            )
        # Check that all relevant axis are included in axis
        if dim != target_shape[ind] and ind not in axis:
            raise ValueError(
                f"{info} {ind}-th array dimension differs from "
                f"target, but {ind} is not included in "
                f"{axis=}."
            )

    # Compute upscaling factors
    upscale_factors = {}
    for ax in axis:
        if (target_shape[ax] % array_shape[ax]) > 0 and not pad_with_zeros:
            raise ValueError(
                "Incommensurable upscale attempt, "
                f"from {array_shape=} to {target_shape=}."
            )
        upscale_factors[ax] = target_shape[ax] // array_shape[ax]
        # Check that this is not downscaling
        if upscale_factors[ax] < 1:
            raise ValueError(info)
    info = f"{info} Upscale factors: {upscale_factors}"

    # Raise a warning if upscaling is non-homogeneous across all axis
    if warn_if_inhomogeneous:
        if len(set(upscale_factors.values())) > 1:
            warnings.warn(f"{info} (inhomogeneous)")

    # Upscale array, via np.repeat
    upscaled_array = array
    for ax in axis:
        upscaled_array = np.repeat(
            upscaled_array, upscale_factors[ax], axis=ax
        )

    # Check that final shape is correct
    if not upscaled_array.shape == target_shape:
        if pad_with_zeros:
            pad_width = []
            for ax in list(range(len(target_shape))):
                missing = target_shape[ax] - upscaled_array.shape[ax]
                if missing < 0 or (missing > 0 and ax not in axis):
                    raise ValueError(
                        f"{info} " "Something wrong during zero-padding"
                    )
                pad_width.append([0, missing])
            upscaled_array = np.pad(
                upscaled_array,
                pad_width=pad_width,
                mode="constant",
                constant_values=0,
            )
            logging.warning(f"{info} {upscaled_array.shape=}.")
            logging.warning(
                f"Padding upscaled_array with zeros with {pad_width=}"
            )
        else:
            raise ValueError(f"{info} {upscaled_array.shape=}.")

    return upscaled_array


def convert_region_to_low_res(
    *,
    highres_region: tuple[slice, ...],
    lowres_shape: tuple[int, ...],
    highres_shape: tuple[int, ...],
) -> tuple[slice, ...]:
    """
    Convert a region defined for a high-resolution array to the corresponding
    region for a low-resolution array.

    Args:
        highres_region: A region of the high-resolution array, defined in a
            form like `(slice(0, 2), slice(1000, 2000), slice(1000, 2000))`.
        highres_shape: The shape of the high-resolution array.
        lowres_shape: The shape of the low-resolution array.

    Returns:
        Region for low-resolution array.
    """

    error_msg = (
        f"Cannot convert {highres_region=}, "
        f"given {lowres_shape=} and {highres_shape=}."
    )

    ndim = len(lowres_shape)
    if len(highres_shape) != ndim:
        raise ValueError(f"{error_msg} Dimension mismatch.")

    # Loop over dimensions to construct lowres_region, after some relevant
    # checks
    lowres_region = []
    for ind, lowres_size in enumerate(lowres_shape):
        # Check that the high-resolution size is not smaller than the
        # low-resolution size
        highres_size = highres_shape[ind]
        if highres_size < lowres_size:
            raise ValueError(
                f"{error_msg} High-res size smaller than low-res size."
            )
        # Check that sizes are commensurate
        if highres_size % lowres_size > 0:
            raise ValueError(
                f"{error_msg} Incommensurable sizes "
                f"{highres_size=} and {lowres_size=}."
            )
        factor = highres_size // lowres_size
        # Convert old_slice's start/stop attributes
        old_slice = highres_region[ind]
        if old_slice.start % factor > 0 or old_slice.stop % factor > 0:
            raise ValueError(
                f"{error_msg} Cannot transform {old_slice=} "
                f"with {factor=}."
            )
        new_slice_start = old_slice.start // factor
        new_slice_stop = old_slice.stop // factor
        new_slice_step = None
        # Covert old_slice's step attribute
        if old_slice.step:
            if old_slice.step % factor > 0:
                raise ValueError(
                    f"{error_msg} Cannot transform {old_slice=} "
                    f"with {factor=}."
                )
            new_slice_step = old_slice.step // factor
        # Append new slice
        lowres_region.append(
            slice(new_slice_start, new_slice_stop, new_slice_step)
        )

    return tuple(lowres_region)

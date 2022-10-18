import warnings
from typing import Iterable

import numpy as np


def upscale_array(
    *,
    target_shape: Iterable[int] = None,
    array=None,
    axis: Iterable[int] = None,
) -> np.ndarray:
    """
    Upscale array along given axis, to match a
    target shape. Upscaling is based on np.repeat.
    """

    array_shape = array.shape
    info = (
        f"Trying to upscale from {array_shape=} to {target_shape=}, "
        f"acting on {axis=}."
    )

    # Default behavior: use all axis
    if axis is None:
        axis = list(range(len(target_shape)))

    if len(array_shape) != len(target_shape):
        raise ValueError(f"{info} Dimensions-number mismatch.")
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
        upscale_factors[ax] = target_shape[ax] // array_shape[ax]
        # Check that this is not downscaling
        if upscale_factors[ax] < 1:
            raise ValueError(info)
    info = f"{info} Upscale factors: {upscale_factors}"

    # Raise a warning if upscaling is non-homogeneous across all axis
    if len(set(upscale_factors)) > 1:
        warnings.warn(info)

    # Upscale array, via np.repeat
    upscaled_array = array
    for ax in axis:
        upscaled_array = np.repeat(
            upscaled_array, upscale_factors[ax], axis=ax
        )

    # Check that final shape is correct
    if not upscaled_array.shape == target_shape:
        raise Exception(f"{info} {upscaled_array.shape=}.")

    return upscaled_array

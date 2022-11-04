"""
Copyright 2022 (C)
    Friedrich Miescher Institute for Biomedical Research and
    University of Zurich

    Original authors:
    Tommaso Comparin <tommaso.comparin@exact-lab.it>

    This file is part of Fractal and was originally developed by eXact lab
    S.r.l.  <exact-lab.it> under contract with Liberali Lab from the Friedrich
    Miescher Institute for Biomedical Research and Pelkmans Lab from the
    University of Zurich.

Function to increase the shape of an array by replicating it
"""
import warnings
from typing import Sequence

import numpy as np


def upscale_array(
    *,
    array,
    target_shape: Sequence[int],
    axis: Sequence[int] = None,
) -> np.ndarray:
    """
    Upscale an array along a given list of axis (through repeated application
    of ``np.repeat``), to match a target shape.

    :param array: the array to be upscaled
    :param target_shape: the shape of the rescaled array
    :param axis: the axis along which to upscale the array (if ``None``, then \
                 all axis are used)
    :returns: upscaled array, with shape ``target_shape``
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
    if len(set(upscale_factors.values())) > 1:
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

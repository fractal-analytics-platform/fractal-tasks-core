from typing import Union

import dask.array as da
import numpy as np


def load_region(
    data_zyx: da.Array,
    region: tuple[slice, slice, slice],
    compute: bool = True,
    return_as_3D: bool = False,
) -> Union[da.Array, np.ndarray]:
    """
    Load a region from a dask array.

    Can handle both 2D and 3D dask arrays as input and return them as is or
    always as a 3D array.

    Args:
        data_zyx: Dask array (2D or 3D).
        region: Region to load, tuple of three slices (ZYX).
        compute: Whether to compute the result. If `True`, returns a numpy
            array. If `False`, returns a dask array.
        return_as_3D: Whether to return a 3D array, even if the input is 2D.

    Returns:
        3D array.
    """

    if len(region) != 3:
        raise ValueError(
            f"In `load_region`, `region` must have three elements "
            f"(given: {len(region)})."
        )

    if len(data_zyx.shape) == 3:
        img = data_zyx[region]
    elif len(data_zyx.shape) == 2:
        img = data_zyx[(region[1], region[2])]
        if return_as_3D:
            img = np.expand_dims(img, axis=0)
    else:
        raise ValueError(
            f"Shape {data_zyx.shape} not supported for `load_region`"
        )
    if compute:
        return img.compute()
    else:
        return img

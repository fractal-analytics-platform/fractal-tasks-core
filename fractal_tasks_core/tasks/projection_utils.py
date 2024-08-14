from enum import Enum
from typing import Any
from typing import Dict

import dask.array as da
import numpy as np


def safe_sum(
    dask_array: da.Array, axis: int = 0, **kwargs: Dict[str, Any]
) -> da.Array:
    """
    Perform a safe sum on a Dask array to avoid overflow, by clipping the
    result of da.sum & casting it to its original dtype.

    Dask.array already correctly handles promotion to uin32 or uint64 when
    necessary internally, but we want to ensure we clip the result.

    Args:
        dask_array (dask.array.Array): The input Dask array.
        axis (int, optional): The axis along which to sum the array.
            Defaults to 0.
        **kwargs: Additional keyword arguments passed to da.sum.

    Returns:
        dask.array.Array: The result of the sum, safely clipped and cast
            back to the original dtype.
    """
    # Handle empty array
    if any(dim == 0 for dim in dask_array.shape):
        return dask_array

    # Determine the original dtype
    original_dtype = dask_array.dtype
    max_value = np.iinfo(original_dtype).max

    # Perform the sum
    result = da.sum(dask_array, axis=axis, **kwargs)

    # Clip the values to the maximum possible value for the original dtype
    result = da.clip(result, 0, max_value)

    # Cast back to the original dtype
    result = result.astype(original_dtype)

    return result


def mean_wrapper(
    dask_array: da.Array, axis: int = 0, **kwargs: Dict[str, Any]
) -> da.Array:
    """
    Perform a da.mean on the dask_array & cast it to its original dtype.

    Without casting, the result can change dtype to e.g. float64

    Args:
        dask_array (dask.array.Array): The input Dask array.
        axis (int, optional): The axis along which to mean the array.
            Defaults to 0.
        **kwargs: Additional keyword arguments passed to da.mean.

    Returns:
        dask.array.Array: The result of the mean, cast back to the original
            dtype.
    """
    # Handle empty array
    if any(dim == 0 for dim in dask_array.shape):
        return dask_array

    # Determine the original dtype
    original_dtype = dask_array.dtype

    # Perform the sum
    result = da.mean(dask_array, axis=axis, **kwargs)

    # Cast back to the original dtype
    result = result.astype(original_dtype)

    return result


class DaskProjectionMethod(Enum):
    """
    Registration method selection

    Choose which method to use for intensity projection along the Z axis.

    Attributes:
        MIP: Maximum intensity projection
        MINIP: Minimum intensityp projection
        MEANIP: Mean intensity projection
        SUMIP: Sum intensityp projection
    """

    MIP = "mip"
    MINIP = "minip"
    MEANIP = "meanip"
    SUMIP = "sumip"

    def apply(
        self, dask_array: da.Array, axis: int = 0, **kwargs: Dict[str, Any]
    ) -> da.Array:
        """
        Apply the selected projection method to the given Dask array.

        Args:
            dask_array (dask.array.Array): The Dask array to project.
            axis (int): The axis along which to apply the projection.
            **kwargs: Additional keyword arguments to pass to the projection
                method.

        Returns:
            dask.array.Array: The resulting Dask array after applying the
                projection.

        Example:
            >>> array = da.random.random((1000, 1000), chunks=(100, 100))
            >>> method = DaskProjectionMethod.MAX
            >>> result = method.apply(array, axis=0)
            >>> computed_result = result.compute()
            >>> print(computed_result)
        """
        # Map the Enum values to the actual Dask array methods
        method_map = {
            DaskProjectionMethod.MIP: lambda arr, axis, **kw: arr.max(
                axis=axis, **kw
            ),
            DaskProjectionMethod.MINIP: lambda arr, axis, **kw: arr.min(
                axis=axis, **kw
            ),
            DaskProjectionMethod.MEANIP: mean_wrapper,
            DaskProjectionMethod.SUMIP: safe_sum,
        }
        # Call the appropriate method, passing in the dask_array explicitly
        return method_map[self](dask_array, axis=axis, **kwargs)

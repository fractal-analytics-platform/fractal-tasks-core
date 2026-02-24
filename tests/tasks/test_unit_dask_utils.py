import dask.array as da
import numpy as np
import pytest

from fractal_tasks_core._projection_utils import (
    max_wrapper,
    mean_wrapper,
    min_wrapper,
    safe_sum,
)

# ---------------------------------------------------------------------------
# safe_sum
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_safe_sum_overflow(dtype: str) -> None:
    max_val = np.iinfo(dtype).max
    array = da.full((2, 5), max_val, chunks=(1, 5), dtype=dtype)
    result = safe_sum(array, axis=0).compute()
    expected = np.full((5,), max_val, dtype=dtype)
    np.testing.assert_array_equal(result, expected)


def test_safe_sum_other_dtype() -> None:
    array = da.full((2, 5), 2**16, chunks=(1, 5), dtype="int32")
    result = safe_sum(array, axis=0).compute()
    expected = np.full((5,), 2**16 * 2, dtype="int32")
    np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# mean_wrapper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["int32", "float64", "uint8", "uint16"])
def test_mean_wrapper_dtypes(dtype: str) -> None:
    array = da.arange(10, dtype=dtype).reshape((2, 5))
    result = mean_wrapper(array, axis=0).compute()
    expected = np.arange(10, dtype=dtype).reshape((2, 5)).mean(axis=0).astype(dtype)
    assert result.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# max_wrapper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["uint8", "uint16", "int32"])
def test_max_wrapper_dtypes(dtype: str) -> None:
    array = da.arange(10, dtype=dtype).reshape((2, 5))
    result = max_wrapper(array, axis=0).compute()
    expected = np.arange(10, dtype=dtype).reshape((2, 5)).max(axis=0)
    assert result.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# min_wrapper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["uint8", "uint16", "int32"])
def test_min_wrapper_dtypes(dtype: str) -> None:
    array = da.arange(10, dtype=dtype).reshape((2, 5))
    result = min_wrapper(array, axis=0).compute()
    expected = np.arange(10, dtype=dtype).reshape((2, 5)).min(axis=0)
    assert result.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(result, expected)

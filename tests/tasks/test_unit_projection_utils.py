import dask.array as da
import numpy as np

from fractal_tasks_core.tasks.projection_utils import mean_wrapper
from fractal_tasks_core.tasks.projection_utils import safe_sum


def test_uint8_no_overflow():
    # Test sum on uint8 array without overflow
    array = da.ones((100, 100), chunks=(50, 50), dtype="uint8")
    result = safe_sum(array).compute()
    expected = np.full((100,), 100, dtype="uint8")
    np.testing.assert_array_equal(result, expected)


def test_uint16_no_overflow():
    # Test sum on uint16 array without overflow
    array = da.ones((100, 100), chunks=(50, 50), dtype="uint16")
    result = safe_sum(array).compute()
    expected = np.full((100,), 100, dtype="uint16")
    np.testing.assert_array_equal(result, expected)


def test_uint8_overflow():
    # Test sum on uint8 array with potential overflow
    array = da.full((1000, 1000), 255, chunks=(500, 500), dtype="uint8")
    result = safe_sum(array, axis=0).compute()
    expected = np.full((1000,), 255, dtype="uint8")  # Should be clipped to 255
    np.testing.assert_array_equal(result, expected)


def test_uint16_overflow():
    # Test sum on uint16 array with potential overflow
    array = da.full((1000, 1000), 65535, chunks=(500, 500), dtype="uint16")
    result = safe_sum(array, axis=0).compute()
    expected = np.full(
        (1000,), 65535, dtype="uint16"
    )  # Should be clipped to 65535
    np.testing.assert_array_equal(result, expected)


def test_other_dtype():
    # Test sum on int32 array (should not cast or clip)
    array = da.full((1000, 1000), 2**16, chunks=(500, 500), dtype="int32")
    result = safe_sum(array, axis=0).compute()
    expected = np.full((1000,), 2**16 * 1000, dtype="int32")
    np.testing.assert_array_equal(result, expected)


def test_empty_array():
    # Test sum on an empty array
    array = da.zeros((0,), chunks=(0,), dtype="uint8")
    result = safe_sum(array, axis=None).compute()
    expected = np.array(0, dtype="uint8")
    np.testing.assert_array_equal(result, expected)


def test_axis_sum():
    # Test sum along a specific axis
    array = da.full((10, 10), 255, chunks=(5, 5), dtype="uint8")
    result = safe_sum(array, axis=1).compute()
    expected = np.full((10,), 255, dtype="uint8")
    np.testing.assert_array_equal(result, expected)


def test_mean_wrapper_int32():
    # Test mean on int32 array, ensuring dtype is preserved
    array = da.arange(10, dtype="int32", chunks=5)
    result = mean_wrapper(array, axis=0).compute()
    expected = np.mean(np.arange(10, dtype="int32")).astype("int32")
    assert result.dtype == "int32"
    np.testing.assert_array_equal(result, expected)


def test_mean_wrapper_float64():
    # Test mean on float64 array, ensuring dtype is preserved
    array = da.arange(10, dtype="float64", chunks=5)
    result = mean_wrapper(array, axis=0).compute()
    expected = np.mean(np.arange(10, dtype="float64"))
    assert result.dtype == "float64"
    np.testing.assert_array_equal(result, expected)


def test_mean_wrapper_uint8():
    # Test mean on uint8 array, ensuring dtype is preserved
    array = da.arange(10, dtype="uint8", chunks=5)
    result = mean_wrapper(array, axis=0).compute()
    expected = np.mean(np.arange(10, dtype="uint8")).astype("uint8")
    assert result.dtype == "uint8"
    np.testing.assert_array_equal(result, expected)


def test_mean_wrapper_uint16():
    # Test mean on uint16 array, ensuring dtype is preserved
    array = da.arange(10, dtype="uint16", chunks=5)
    result = mean_wrapper(array, axis=0).compute()
    expected = np.mean(np.arange(10, dtype="uint16")).astype("uint16")
    assert result.dtype == "uint16"
    np.testing.assert_array_equal(result, expected)


def test_mean_wrapper_empty_array():
    # Test mean on an empty array
    array = da.zeros((0,), chunks=(0,), dtype="uint8")
    result = mean_wrapper(array).compute()
    expected = np.array([], dtype="uint8")
    assert result.dtype == "uint8"
    np.testing.assert_array_equal(result, expected)


def test_mean_wrapper_axis():
    # Test mean along a specific axis
    array = da.arange(20, dtype="int32", chunks=(5,)).reshape((4, 5))
    result = mean_wrapper(array, axis=1).compute()
    expected = np.mean(
        np.arange(20, dtype="int32").reshape((4, 5)), axis=1
    ).astype("int32")
    assert result.dtype == "int32"
    np.testing.assert_array_equal(result, expected)


def test_mean_wrapper_with_kwargs():
    # Test mean with additional kwargs (e.g., keepdims)
    array = da.arange(20, dtype="int32", chunks=(5,)).reshape((4, 5))
    result = mean_wrapper(array, axis=1, keepdims=True).compute()
    expected = np.mean(
        np.arange(20, dtype="int32").reshape((4, 5)), axis=1, keepdims=True
    ).astype("int32")
    assert result.dtype == "int32"
    np.testing.assert_array_equal(result, expected)

import dask.array as da
import numpy as np

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

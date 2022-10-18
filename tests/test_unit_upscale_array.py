from typing import Iterable

import numpy as np
import pytest

from fractal_tasks_core.lib_upscale_array import upscale_array


list_success = []
list_success.append(((1, 2, 3), (1, 2, 3), [0, 1, 2]))
list_success.append(((1, 2, 3), (1, 2, 3), [0, 1]))
list_success.append(((1, 2, 3), (1, 2, 3), [0]))
list_success.append(((1, 2, 3), (1, 2, 3), None))
list_success.append(((1, 2, 3), (1, 4, 6), [1, 2]))
list_success.append(((1, 2, 3), (1, 4, 12), [1, 2]))


@pytest.mark.parametrize("old_shape,target_shape,axis", list_success)
def test_upscale_array_success(
    old_shape: Iterable[int], target_shape: Iterable[int], axis: Iterable[int]
):
    old_array = np.ones(old_shape)
    new_array = upscale_array(
        target_shape=target_shape, array=old_array, axis=axis
    )
    assert new_array.shape == target_shape


list_fail = []
list_fail.append(((1, 2, 3, 4), (1, 2, 6), [0, 1]))
list_fail.append(((1, 2, 3, 4), (1, 2, 6), [0, -2]))
list_fail.append(((1, 4), (1, 2), None))
list_fail.append(((1, 4), (1, 2), [0]))


@pytest.mark.parametrize("old_shape,target_shape,axis", list_fail)
def test_upscale_array_fail(
    old_shape: Iterable[int], target_shape: Iterable[int], axis: Iterable[int]
):
    old_array = np.ones(old_shape)
    with pytest.raises(ValueError):
        upscale_array(target_shape=target_shape, array=old_array, axis=axis)

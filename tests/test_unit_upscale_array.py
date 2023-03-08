from typing import List
from typing import Sequence
from typing import Tuple

import numpy as np
import pytest
from devtools import debug

from fractal_tasks_core.lib_upscale_array import convert_region_to_low_res
from fractal_tasks_core.lib_upscale_array import upscale_array


list_success: List[Tuple] = []
list_success.append(((1, 2, 3), (1, 2, 3), [0, 1, 2]))
list_success.append(((1, 2, 3), (1, 2, 3), [0, 1]))
list_success.append(((1, 2, 3), (1, 2, 3), [0]))
list_success.append(((1, 2, 3), (1, 2, 3), None))
list_success.append(((1, 2, 3), (1, 4, 6), [1, 2]))
list_success.append(((1, 2, 3), (1, 4, 12), [1, 2]))


@pytest.mark.parametrize("old_shape,target_shape,axis", list_success)
def test_upscale_array_success(
    old_shape: Sequence[int], target_shape: Sequence[int], axis: Sequence[int]
):
    old_array = np.ones(old_shape)
    new_array = upscale_array(
        target_shape=target_shape, array=old_array, axis=axis
    )
    assert new_array.shape == target_shape


list_fail: List[Tuple] = []
list_fail.append(((1, 2, 3, 4), (1, 2, 6), [0, 1]))
list_fail.append(((1, 2, 3, 4), (1, 2, 6), [0, -2]))
list_fail.append(((1, 4), (1, 2), None))
list_fail.append(((1, 4), (1, 2), [0]))


@pytest.mark.parametrize("old_shape,target_shape,axis", list_fail)
def test_upscale_array_fail(
    old_shape: Sequence[int], target_shape: Sequence[int], axis: Sequence[int]
):
    old_array = np.ones(old_shape)
    with pytest.raises(ValueError):
        upscale_array(target_shape=target_shape, array=old_array, axis=axis)


def test_incommensurable_upscaling():
    """
    GIVEN an array with shape incommensurable with respect to a target shape
    WHEN calling upscale_array
    THEN
      * Fail as expected, if pad_with_zeros=False
      * Return an upscaled array with the correct (target) shape otherwise
    """
    array = np.ones((3, 2))
    target_shape = (5, 5)

    # If fail_on_mismatch is not explicitly False, fail
    with pytest.raises(ValueError):
        upscaled_array = upscale_array(array=array, target_shape=target_shape)

    # When fail_on_mismatch=False, the array should have the right shape
    upscaled_array = upscale_array(
        array=array,
        target_shape=target_shape,
        pad_with_zeros=True,
    )
    assert upscaled_array.shape == target_shape
    debug(upscaled_array)


def test_convert_region_to_low_res():

    # Successful conversion
    highres_shape = (8,)
    lowres_shape = (4,)
    highres_region = (slice(2, 6),)
    expected_lowres_region = (slice(1, 3),)
    lowres_region = convert_region_to_low_res(
        highres_shape=highres_shape,
        lowres_shape=lowres_shape,
        highres_region=highres_region,
    )
    debug(highres_region)
    debug(lowres_region)
    assert lowres_region == expected_lowres_region

    # Conversion in the wrong direction
    with pytest.raises(ValueError):
        convert_region_to_low_res(
            highres_shape=lowres_shape,
            lowres_shape=highres_shape,
            highres_region=highres_region,
        )

    # Incommensurability error (1/2)
    with pytest.raises(ValueError):
        highres_shape = (9,)
        lowres_shape = (4,)
        highres_region = (slice(2, 6),)
        convert_region_to_low_res(
            highres_shape=highres_shape,
            lowres_shape=lowres_shape,
            highres_region=highres_region,
        )

    # Incommensurability error (2/2)
    with pytest.raises(ValueError):
        highres_shape = (8,)
        lowres_shape = (4,)
        highres_region = (slice(3, 7),)
        convert_region_to_low_res(
            highres_shape=highres_shape,
            lowres_shape=lowres_shape,
            highres_region=highres_region,
        )

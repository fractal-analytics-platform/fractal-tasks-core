import numpy as np
import pytest

from fractal_tasks_core.tasks.cellpose_transforms import (
    CellposeCustomNormalizer,
)
from fractal_tasks_core.tasks.cellpose_transforms import (
    normalized_img,
)


@pytest.mark.parametrize(
    "default_normalize, lower_percentile, upper_percentile, lower_bound, "
    "upper_bound, expected_value_error",
    [
        (True, None, None, None, None, False),
        (True, 1, 99, None, None, True),
        (True, 1, None, None, None, True),
        (True, None, 99, None, None, True),
        (True, 1, 99, 0, 100, True),
        (True, None, None, 0, 100, True),
        (True, None, None, None, 100, True),
        (True, None, None, 0, None, True),
        (False, None, None, None, None, False),
        (False, 1, 99, None, None, False),
        (False, 1, None, None, None, True),
        (False, None, 99, None, None, True),
        (False, 1, 99, 0, 100, True),
        (False, None, None, 0, 100, False),
        (False, None, None, None, 100, True),
        (False, None, None, 0, None, True),
    ],
)
def test_CellposeCustomNormalizer(
    default_normalize,
    lower_percentile,
    upper_percentile,
    lower_bound,
    upper_bound,
    expected_value_error,
):
    if expected_value_error:
        pass
        with pytest.raises(ValueError):
            CellposeCustomNormalizer(
                default_normalize=default_normalize,
                lower_percentile=lower_percentile,
                upper_percentile=upper_percentile,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            )
    else:
        assert (
            CellposeCustomNormalizer(
                default_normalize=default_normalize,
                lower_percentile=lower_percentile,
                upper_percentile=upper_percentile,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            ).default_normalize
            == default_normalize
        )


def test_normalized_img_percentile():
    # Create a 4D numpy array with values evenly distributed from 0 to 1000
    img = np.linspace(1, 1000, num=1000).reshape((1, 10, 10, 10))

    # Normalize the image
    normalized = normalized_img(img, axis=0, lower_p=1.0, upper_p=99.0)

    # Check dimensions are unchanged
    assert (
        img.shape == normalized.shape
    ), "Normalized image should have the same shape as input"

    # Check type is float32 as per the function's specification
    assert (
        normalized.dtype == np.float32
    ), "Normalized image should be of type float32"

    # Check that the normalization results in the expected clipping
    assert np.sum(np.sum(np.sum(np.sum(normalized <= 0)))) == 10
    assert np.sum(np.sum(np.sum(np.sum(normalized >= 1)))) == 10


@pytest.mark.parametrize(
    "lower_bound, upper_bound, lower_than_0, higher_than_1",
    [
        (0, 901, 0, 100),
        (100, 901, 100, 100),
        (10, 991, 10, 10),
        (1, 999, 1, 2),
    ],
)
def test_normalized_img_bounds(
    lower_bound, upper_bound, lower_than_0, higher_than_1
):
    # Create a 4D numpy array with values evenly distributed from 0 to 1000
    img = np.linspace(1, 1000, num=1000).reshape((1, 10, 10, 10))

    # Normalize the image
    normalized = normalized_img(
        img,
        axis=0,
        lower_p=None,
        upper_p=None,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )

    # Check dimensions are unchanged
    assert (
        img.shape == normalized.shape
    ), "Normalized image should have the same shape as input"

    # Check type is float32 as per the function's specification
    assert (
        normalized.dtype == np.float32
    ), "Normalized image should be of type float32"

    # Check that the normalization results in the expected clipping
    assert np.sum(np.sum(np.sum(np.sum(normalized <= 0)))) == lower_than_0
    assert np.sum(np.sum(np.sum(np.sum(normalized >= 1)))) == higher_than_1

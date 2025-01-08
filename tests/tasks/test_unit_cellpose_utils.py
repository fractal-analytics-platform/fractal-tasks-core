import numpy as np
import pytest

from fractal_tasks_core.tasks.cellpose_utils import (
    _normalize_cellpose_channels,
)
from fractal_tasks_core.tasks.cellpose_utils import (
    CellposeCustomNormalizer,
)
from fractal_tasks_core.tasks.cellpose_utils import (
    normalized_img,
)


@pytest.mark.parametrize(
    "type, lower_percentile, upper_percentile, lower_bound, "
    "upper_bound, expected_value_error",
    [
        ("default", None, None, None, None, False),
        ("default", 1, 99, None, None, True),
        ("default", 1, None, None, None, True),
        ("default", None, 99, None, None, True),
        ("default", 1, 99, 0, 100, True),
        ("default", None, None, 0, 100, True),
        ("default", None, None, None, 100, True),
        ("default", None, None, 0, None, True),
        ("no_normalization", None, None, None, None, False),
        ("custom", 1, 99, None, None, False),
        ("custom", 1, None, None, None, True),
        ("custom", None, 99, None, None, True),
        ("custom", 1, 99, 0, 100, True),
        ("custom", None, None, 0, 100, False),
        ("custom", None, None, None, 100, True),
        ("custom", None, None, 0, None, True),
        ("wrong_type", None, None, None, None, True),
    ],
)
def test_CellposeCustomNormalizer(
    type,
    lower_percentile,
    upper_percentile,
    lower_bound,
    upper_bound,
    expected_value_error,
):
    if expected_value_error:
        with pytest.raises(ValueError):
            CellposeCustomNormalizer(
                type=type,
                lower_percentile=lower_percentile,
                upper_percentile=upper_percentile,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            )
    else:
        if type == "default":
            assert CellposeCustomNormalizer(
                type=type,
                lower_percentile=lower_percentile,
                upper_percentile=upper_percentile,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            ).cellpose_normalize
        else:
            assert not (
                CellposeCustomNormalizer(
                    type=type,
                    lower_percentile=lower_percentile,
                    upper_percentile=upper_percentile,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                ).cellpose_normalize
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
def test_normalize_cellpose_channels(
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


def test_normalized_img_bounds_single_channel():
    # Create a 4D numpy array with values evenly distributed from 0 to 1000
    # Single channel image
    x = np.linspace(1, 1000, num=1000).reshape((1, 10, 10, 10))
    channels = [0, 0]

    # No normalization
    normalize_default = CellposeCustomNormalizer()
    x_norm = _normalize_cellpose_channels(
        x=np.copy(x),
        channels=channels,
        normalize=normalize_default,
        normalize2=normalize_default,
    )
    assert (x == x_norm).all()

    # Custom normalization
    normalize_custom_c1 = CellposeCustomNormalizer(
        type="custom", lower_bound=10, upper_bound=991
    )
    x_norm = _normalize_cellpose_channels(
        x=np.copy(x),
        channels=channels,
        normalize=normalize_custom_c1,
        normalize2=normalize_default,
    )
    # Check that the normalization results in the expected ranges
    assert np.sum(np.sum(np.sum(np.sum(x_norm <= 0)))) == 10
    assert np.sum(np.sum(np.sum(np.sum(x_norm >= 1)))) == 10


def test_normalized_img_bounds_dual_channel():
    # Create a 4D numpy array with values evenly distributed from 0 to 2000
    # Dual channel image
    x = np.linspace(1, 2000, num=2000).reshape((2, 10, 10, 10))
    channels = [1, 2]

    # Run as default => applies no normalization
    normalize_default = CellposeCustomNormalizer()

    # Normalize the image
    x_norm = _normalize_cellpose_channels(
        x=np.copy(x),
        channels=channels,
        normalize=normalize_default,
        normalize2=normalize_default,
    )
    assert (x == x_norm).all()

    # Run with no normalization option => applies no normalization
    normalize_no = CellposeCustomNormalizer(
        type="no_normalization",
    )

    # Normalize the image
    x_norm = _normalize_cellpose_channels(
        x=np.copy(x),
        channels=channels,
        normalize=normalize_no,
        normalize2=normalize_no,
    )
    assert (x == x_norm).all()

    # Use 1 as default and the other as custom => rais Value error
    normalize_custom = CellposeCustomNormalizer(
        type="custom", lower_bound=10, upper_bound=990
    )
    with pytest.raises(ValueError):
        _normalize_cellpose_channels(
            x=x,
            channels=channels,
            normalize=normalize_default,
            normalize2=normalize_custom,
        )
    with pytest.raises(ValueError):
        _normalize_cellpose_channels(
            x=x,
            channels=channels,
            normalize=normalize_custom,
            normalize2=normalize_default,
        )

    # Use a custom normalizer for both channels.
    # Channel 1 goes from 1 to 1000, channel 2 from 1001 to 2000
    normalize_custom_c1 = CellposeCustomNormalizer(
        type="custom", lower_bound=10, upper_bound=991
    )
    normalize_custom_c2 = CellposeCustomNormalizer(
        type="custom", lower_bound=1010, upper_bound=1991
    )
    x_norm = _normalize_cellpose_channels(
        x=np.copy(x),
        channels=channels,
        normalize=normalize_custom_c1,
        normalize2=normalize_custom_c2,
    )

    # Check that the normalization results in the expected ranges
    assert np.sum(np.sum(np.sum(np.sum(x_norm <= 0)))) == 20
    assert np.sum(np.sum(np.sum(np.sum(x_norm >= 1)))) == 20

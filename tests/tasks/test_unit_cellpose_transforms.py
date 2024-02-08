import pytest

from fractal_tasks_core.tasks.cellpose_transforms import (
    CellposeCustomNormalizer,
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

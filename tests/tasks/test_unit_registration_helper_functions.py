import numpy as np
import pytest
from devtools import debug

from fractal_tasks_core.tasks.calculate_2D_registration_image_based import (
    calculate_physical_shifts,
)  # noqa


@pytest.mark.parametrize(
    "shifts",
    [
        np.array([10.0]),
        np.array([10.0, 20.0]),
        np.array([10.0, 20.0, 30.0]),
    ],
)
def test_calculate_physical_shifts(shifts):
    level = 1
    coarsening_xy = 2
    full_res_pxl_sizes_zyx = np.array([3.0, 4.0, 5.0])
    factors = coarsening_xy**level * full_res_pxl_sizes_zyx
    debug(shifts)

    if len(shifts) not in [2, 3]:
        with pytest.raises(ValueError):
            calculate_physical_shifts(
                shifts, level, coarsening_xy, full_res_pxl_sizes_zyx
            )
        return

    elif len(shifts) == 2:
        expected_shifts_physical = [0.0] + list(shifts * factors[1:])
    elif len(shifts) == 3:
        expected_shifts_physical = shifts * factors

    shifts_physical = calculate_physical_shifts(
        shifts,
        level,
        coarsening_xy,
        full_res_pxl_sizes_zyx,
    )
    debug(shifts_physical)
    assert np.allclose(shifts_physical, expected_shifts_physical)

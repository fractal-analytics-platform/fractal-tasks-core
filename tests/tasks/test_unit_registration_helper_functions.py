import anndata as ad
import numpy as np
import pytest
from devtools import debug

from fractal_tasks_core.tasks.calculate_2D_registration_image_based import (
    calculate_physical_shifts,
)
from fractal_tasks_core.tasks.calculate_2D_registration_image_based import (
    get_ROI_table_with_translation,
)


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


@pytest.mark.parametrize("fail", [False, True])
def test_get_ROI_table_with_translation(fail: bool):
    new_shifts = {
        "FOV_1": [
            0,
            7.8,
            32.5,
        ],
        "FOV_2": [
            0,
            7.8,
            32.5,
        ],
    }
    if fail:
        new_shifts.pop("FOV_2")
    ROI_table = ad.AnnData(
        X=np.array(
            [
                [
                    -1.4483e03,
                    -1.5177e03,
                    0.0000e00,
                    4.1600e02,
                    3.5100e02,
                    1.0000e00,
                    -1.4483e03,
                    -1.5177e03,
                ],
                [
                    -1.0323e03,
                    -1.5177e03,
                    0.0000e00,
                    4.1600e02,
                    3.5100e02,
                    1.0000e00,
                    -1.0323e03,
                    -1.5177e03,
                ],
            ],
            dtype=np.float32,
        )
    )
    ROI_table.obs_names = ["FOV_1", "FOV_2"]
    ROI_table.var_names = [
        "x_micrometer",
        "y_micrometer",
        "z_micrometer",
        "len_x_micrometer",
        "len_y_micrometer",
        "len_z_micrometer",
        "x_micrometer_original",
        "y_micrometer_original",
    ]
    if fail:
        with pytest.raises(ValueError) as e:
            get_ROI_table_with_translation(ROI_table, new_shifts)
        debug(e.value)
        assert "different length" in str(e.value)
    else:
        get_ROI_table_with_translation(ROI_table, new_shifts)

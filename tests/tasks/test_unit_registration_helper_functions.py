import anndata as ad
import numpy as np
import pytest
from devtools import debug

from fractal_tasks_core.tasks.apply_registration_to_ROI_tables import (
    add_zero_translation_columns,
)
from fractal_tasks_core.tasks.apply_registration_to_ROI_tables import (
    get_acquisition_paths,
)
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


def test_get_acquisition_paths():

    # Successful call
    image_1 = dict(path="path1", acquisition=1)
    image_2 = dict(path="path2", acquisition=2)
    zattrs = dict(well=dict(images=[image_1, image_2]))
    res = get_acquisition_paths(zattrs)
    debug(res)
    assert res == {1: "path1", 2: "path2"}

    # Fail (missing acquisition key)
    image_1 = dict(path="path1", acquisition=1)
    image_2 = dict(path="path2")
    zattrs = dict(well=dict(images=[image_1, image_2]))
    with pytest.raises(ValueError):
        get_acquisition_paths(zattrs)

    # Fail (non-unique acquisition value)
    image_1 = dict(path="path1", acquisition=1)
    image_2 = dict(path="path2", acquisition=1)
    zattrs = dict(well=dict(images=[image_1, image_2]))
    with pytest.raises(NotImplementedError):
        get_acquisition_paths(zattrs)


def test_add_zero_translation_columns():
    # Run successfully
    adata = ad.AnnData(X=np.ones((4, 2)))
    adata.var_names = ["old_column_A", "old_column_B"]
    new_adata = add_zero_translation_columns(adata)
    debug(new_adata.X)
    for ax in ["x", "y", "z"]:
        column = f"translation_{ax}"
        assert column in new_adata.var_names
        assert np.allclose(new_adata.to_df()[column], 0.0)
    # Fail because of one column already present
    adata = ad.AnnData(X=np.ones((4, 2)))
    adata.var_names = ["old_column_A", "translation_x"]
    with pytest.raises(ValueError) as e:
        add_zero_translation_columns(adata)
    debug(e.value)
    assert "roi table already contains translation columns" in str(e.value)

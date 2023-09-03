import anndata as ad
import numpy as np
import pandas as pd
import pytest
from devtools import debug

from fractal_tasks_core.tasks.apply_registration_to_ROI_tables import (
    add_zero_translation_columns,
)
from fractal_tasks_core.tasks.apply_registration_to_ROI_tables import (
    apply_registration_to_single_ROI_table,
)
from fractal_tasks_core.tasks.apply_registration_to_ROI_tables import (
    calculate_min_max_across_dfs,
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


@pytest.fixture
def translation_table():
    # Dummy table with calculated shifts per ROI (for 2 ROIs)
    return [
        pd.DataFrame(
            {
                "translation_z": [0, 0],
                "translation_y": [4, 5],
                "translation_x": [9, 11],
            },
            index=["FOV_1", "FOV_2"],
        ),
        pd.DataFrame(
            {
                "translation_z": [0, 0],
                "translation_y": [5, 6],
                "translation_x": [-5, 7],
            },
            index=["FOV_1", "FOV_2"],
        ),
        pd.DataFrame(
            {
                "translation_z": [0, 0],
                "translation_y": [6, -3],
                "translation_x": [-1, 12],
            },
            index=["FOV_1", "FOV_2"],
        ),
    ]


@pytest.fixture
def max_df_exp():
    return pd.DataFrame(
        {
            "translation_z": [0, 0],
            "translation_y": [6, 6],
            "translation_x": [9, 12],
        },
        index=["FOV_1", "FOV_2"],
    )


@pytest.fixture
def min_df_exp():
    return pd.DataFrame(
        {
            "translation_z": [0, 0],
            "translation_y": [4, -3],
            "translation_x": [-5, 7],
        },
        index=["FOV_1", "FOV_2"],
    )


@pytest.fixture
def roi_table():
    return ad.AnnData(
        pd.DataFrame(
            {
                "x_micrometer": [-1439.300049, -1020.300049],
                "y_micrometer": [-1517.699951, -1517.699951],
                "z_micrometer": [0.0, 0.0],
                "len_x_micrometer": [416.0, 416.0],
                "len_y_micrometer": [351.0, 351.0],
                "len_z_micrometer": [0.0, 0.0],
            },
            index=["FOV_1", "FOV_2"],
        )
    )


translated_ROI_table_df = pd.DataFrame(
    {
        "x_micrometer": [-1430.300049, -1008.300049],
        "y_micrometer": [-1511.699951, -1511.699951],
        "z_micrometer": [0.0, 0.0],
        "len_x_micrometer": [402.0, 411.0],
        "len_y_micrometer": [349.0, 342.0],
        "len_z_micrometer": [0.0, 0.0],
        "translation_z": [0.0, 0.0],
        "translation_y": [0.0, 0.0],
        "translation_x": [0.0, 0.0],
    },
    index=["FOV_1", "FOV_2"],
)


def test_calculate_min_max_across_dfs(
    translation_table, max_df_exp, min_df_exp
):
    max_df, min_df = calculate_min_max_across_dfs(translation_table)
    assert (max_df == max_df_exp).all().all()
    assert (min_df == min_df_exp).all().all()


def test_apply_registration_to_single_ROI_table(roi_table, translation_table):
    adata_table = add_zero_translation_columns(roi_table)
    max_df, min_df = calculate_min_max_across_dfs(translation_table)
    registered_table = apply_registration_to_single_ROI_table(
        adata_table, max_df, min_df
    ).to_df()
    assert (registered_table == translated_ROI_table_df).all().all()


def test_failure_apply_registration_to_single_ROI_table(
    roi_table, translation_table
):
    adata_table = add_zero_translation_columns(roi_table)
    max_df, min_df = calculate_min_max_across_dfs(translation_table)
    max_df = pd.DataFrame(
        {
            "translation_z": [0, 0, 0],
            "translation_y": [6, 6, 6],
            "translation_x": [9, 12, 12],
        },
        index=["FOV_1", "FOV_2", "Fov_3"],
    )
    with pytest.raises(ValueError):
        apply_registration_to_single_ROI_table(adata_table, max_df, min_df)

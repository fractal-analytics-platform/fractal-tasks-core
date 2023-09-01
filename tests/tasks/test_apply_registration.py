import anndata as ad
import numpy as np
import pandas as pd
import pytest

from fractal_tasks_core.tasks.apply_registration_to_ROI_tables import (
    add_zero_translation_columns,
)
from fractal_tasks_core.tasks.apply_registration_to_ROI_tables import (
    apply_registration_to_single_ROI_table,
)
from fractal_tasks_core.tasks.apply_registration_to_ROI_tables import (
    calculate_min_max_across_dfs,
)


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


def test_add_translation_columns(roi_table):
    adata_table = add_zero_translation_columns(roi_table)
    assert adata_table.shape == (2, 9)
    assert np.sum(np.sum(adata_table.X[:, -3:])) == 0


def test_adding_translation_columns_fails_if_available(roi_table):
    adata_table = add_zero_translation_columns(roi_table)
    with pytest.raises(ValueError) as exc_info:
        add_zero_translation_columns(adata_table)

    assert (
        str(exc_info.value)
        == "The roi table already contains translation columns. "
        "Did you enter a wrong reference cycle?"
    )


def test_apply_registration_to_single_ROI_table(roi_table, translation_table):
    adata_table = add_zero_translation_columns(roi_table)
    max_df, min_df = calculate_min_max_across_dfs(translation_table)
    registered_table = apply_registration_to_single_ROI_table(
        adata_table, max_df, min_df, rois=["FOV_1", "FOV_2"]
    ).to_df()
    assert (registered_table == translated_ROI_table_df).all().all()

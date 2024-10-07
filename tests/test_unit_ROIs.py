from pathlib import Path

import anndata as ad
import dask.array as da
import numpy as np
import pandas as pd
import pytest
import zarr
from anndata._io.specs import write_elem
from devtools import debug

from fractal_tasks_core.roi import (
    are_ROI_table_columns_valid,
)
from fractal_tasks_core.roi import (
    array_to_bounding_box_table,
)
from fractal_tasks_core.roi import check_valid_ROI_indices
from fractal_tasks_core.roi import (
    convert_indices_to_regions,
)
from fractal_tasks_core.roi import (
    convert_ROI_table_to_indices,
)
from fractal_tasks_core.roi import (
    convert_ROIs_from_3D_to_2D,
)
from fractal_tasks_core.roi import empty_bounding_box_table
from fractal_tasks_core.roi import (
    find_overlaps_in_ROI_indices,
)
from fractal_tasks_core.roi import get_image_grid_ROIs
from fractal_tasks_core.roi import get_single_image_ROI
from fractal_tasks_core.roi import is_ROI_table_valid
from fractal_tasks_core.roi import (
    is_standard_roi_table,
)
from fractal_tasks_core.roi import load_region
from fractal_tasks_core.roi import prepare_FOV_ROI_table
from fractal_tasks_core.roi import prepare_well_ROI_table
from fractal_tasks_core.roi import reset_origin
from fractal_tasks_core.roi.v1 import create_roi_table_from_df_list


PIXEL_SIZE_X = 0.1625
PIXEL_SIZE_Y = 0.1625
PIXEL_SIZE_Z = 1.0

IMG_SIZE_X = 2560
IMG_SIZE_Y = 2160
NUM_Z_PLANES = 4

FOV_IDS = ["1", "2", "7", "9"]
FOV_NAMES = [f"FOV_{ID}" for ID in FOV_IDS]


def get_metadata_dataframe():
    """
    Create artificial metadata dataframe
    """
    df = pd.DataFrame(np.zeros((4, 13)), dtype=float)
    df.index = FOV_IDS
    df.columns = [
        "x_micrometer",
        "y_micrometer",
        "z_micrometer",
        "x_micrometer_original",
        "y_micrometer_original",
        "x_pixel",
        "y_pixel",
        "z_pixel",
        "pixel_size_x",
        "pixel_size_y",
        "pixel_size_z",
        "bit_depth",
        "time",
    ]
    img_size_x_micrometer = IMG_SIZE_X * PIXEL_SIZE_X
    img_size_y_micrometer = IMG_SIZE_Y * PIXEL_SIZE_Y
    df["x_micrometer"] = [
        0.0,
        img_size_x_micrometer,
        0.0,
        img_size_x_micrometer,
    ]
    df["y_micrometer"] = [
        0.0,
        0.0,
        img_size_y_micrometer,
        img_size_y_micrometer,
    ]
    df["x_micrometer_original"] = df["x_micrometer"]
    df["y_micrometer_original"] = df["y_micrometer"]
    df["z_micrometer"] = [0.0, 0.0, 0.0, 0.0]
    df["x_pixel"] = [IMG_SIZE_X] * 4
    df["y_pixel"] = [IMG_SIZE_Y] * 4
    df["z_pixel"] = [NUM_Z_PLANES] * 4
    df["pixel_size_x"] = [PIXEL_SIZE_X] * 4
    df["pixel_size_y"] = [PIXEL_SIZE_Y] * 4
    df["pixel_size_z"] = [PIXEL_SIZE_Z] * 4
    df["bit_depth"] = [16.0] * 4
    df["time"] = "2020-08-12 15:36:36.234000+0000"

    return df


list_pxl_sizes = []
list_pxl_sizes.append([PIXEL_SIZE_Z, PIXEL_SIZE_Y, PIXEL_SIZE_X])
list_pxl_sizes.append([val + 1e-6 for val in list_pxl_sizes[0]])
list_pxl_sizes.append([val - 1e-6 for val in list_pxl_sizes[0]])


list_level_coarsening = [
    (0, 2),
    (1, 2),
    (2, 2),
    (3, 2),
    (0, 3),
    (1, 3),
    (2, 3),
    (3, 3),
    (0, 7),
    (1, 7),
    (2, 7),
]

list_params = []
for pxl_sizes in list_pxl_sizes:
    for level_coarsening in list_level_coarsening:
        level, coarsening = level_coarsening[:]
        list_params.append((level, coarsening, pxl_sizes))


@pytest.mark.parametrize(
    "level,coarsening_xy,full_res_pxl_sizes_zyx", list_params
)
def test_ROI_indices_3D(level, coarsening_xy, full_res_pxl_sizes_zyx):
    metadata_dataframe = get_metadata_dataframe()
    adata = prepare_FOV_ROI_table(metadata_dataframe)
    assert list(adata.obs_names) == FOV_NAMES

    list_indices = convert_ROI_table_to_indices(
        adata,
        level=level,
        coarsening_xy=coarsening_xy,
        full_res_pxl_sizes_zyx=full_res_pxl_sizes_zyx,
    )
    print()
    original_shape = (
        NUM_Z_PLANES,
        2 * IMG_SIZE_Y,
        2 * IMG_SIZE_X // coarsening_xy**level,
    )
    expected_shape = (
        NUM_Z_PLANES,
        2 * IMG_SIZE_Y // coarsening_xy**level,
        2 * IMG_SIZE_X // coarsening_xy**level,
    )
    print(f"Pixel sizes: {full_res_pxl_sizes_zyx}")
    print(f"Original shape: {original_shape}")
    print(f"coarsening_xy={coarsening_xy}, level={level}")
    print(f"Expected shape: {expected_shape}")
    print("FOV-ROI indices:")
    for indices in list_indices:
        print(indices)
    print()

    assert list_indices[0][5] == list_indices[1][4]
    assert list_indices[0][3] == list_indices[2][2]
    assert (
        abs(
            (list_indices[0][5] - list_indices[0][4])
            - (list_indices[1][5] - list_indices[1][4])
        )
        < coarsening_xy
    )
    assert (
        abs(
            (list_indices[0][3] - list_indices[0][2])
            - (list_indices[1][3] - list_indices[1][2])
        )
        < coarsening_xy
    )
    assert abs(list_indices[1][5] - expected_shape[2]) < coarsening_xy
    assert abs(list_indices[2][3] - expected_shape[1]) < coarsening_xy
    for indices in list_indices:
        assert indices[0] == 0
        assert indices[1] == NUM_Z_PLANES


@pytest.mark.parametrize(
    "level,coarsening_xy,full_res_pxl_sizes_zyx", list_params
)
def test_ROI_indices_2D(level, coarsening_xy, full_res_pxl_sizes_zyx):
    metadata_dataframe = get_metadata_dataframe()
    adata = prepare_FOV_ROI_table(metadata_dataframe)
    adata = convert_ROIs_from_3D_to_2D(adata, PIXEL_SIZE_Z)

    list_indices = convert_ROI_table_to_indices(
        adata,
        level=level,
        coarsening_xy=coarsening_xy,
        full_res_pxl_sizes_zyx=full_res_pxl_sizes_zyx,
    )

    for indices in list_indices:
        assert indices[0] == 0
        assert indices[1] == 1


def test_prepare_well_ROI_table(testdata_path: Path):
    """
    See
    https://github.com/fractal-analytics-platform/fractal-tasks-core/issues/243
    """
    big_df = pd.read_csv(str(testdata_path / "site_metadata_x_pos_bug.csv"))
    well_ids = big_df.well_id.unique()
    for well_id in well_ids:
        debug(well_id)
        # Select a specific well from big_df
        df = big_df.loc[big_df["well_id"] == well_id, :].copy()
        # Construct the AnnData tables for FOVs and for the whole well
        table_FOVs = prepare_FOV_ROI_table(df)
        table_well = prepare_well_ROI_table(df)
        # Check that the well table has a single row
        assert table_well.shape[0] == 1
        # Check that the minima of the first three columns (x/y/z min
        # positions) for the well and for the FOVs are the same
        # NOTE this assumes that columns are sorted in a specific way, we will
        # have to adapt the test otherwise
        for ind in [0, 1, 2]:
            assert (
                abs(min(table_FOVs.X[:, ind]) - min(table_well.X[:, ind]))
                < 1e-12
            )


def test_overlaps_in_indices():
    list_indices = [
        [0, 1, 100, 200, 1000, 2000],
        [0, 1, 200, 300, 1000, 2000],
    ]
    res = find_overlaps_in_ROI_indices(list_indices)
    debug(res)
    assert res is None

    list_indices = [
        [0, 1, 100, 201, 1000, 2000],
        [0, 1, 200, 300, 1000, 2000],
    ]
    res = find_overlaps_in_ROI_indices(list_indices)
    debug(res)
    assert res == (1, 0)

    list_indices = [
        [0, 1, 100, 200, 1000, 2000],
        [0, 1, 200, 300, 1000, 2000],
        [0, 1, 200, 300, 2000, 3000],
        [1, 2, 200, 300, 2000, 3000],
        [1, 2, 299, 400, 2999, 4000],
    ]
    res = find_overlaps_in_ROI_indices(list_indices)
    debug(res)
    assert res == (4, 3)


def test_empty_ROI_table():
    """
    When providing an empty ROI AnnData table to convert_ROI_table_to_indices,
    the resulting indices must be an empty list.
    """
    empty_ROI_table = ad.AnnData(X=None)
    debug(empty_ROI_table)
    indices = convert_ROI_table_to_indices(
        empty_ROI_table,
        full_res_pxl_sizes_zyx=[1.0, 1.0, 1.0],
    )
    assert indices == []


def test_negative_indices():
    """
    Fail as expected when some index is negative.
    """
    # Write valid table to a zarr group
    columns = EXPECTED_COLUMNS.copy()
    ROI_table = ad.AnnData(np.array([[0.5, -0.8, 0.5, 10, 10, 10]]))
    ROI_table.var_names = columns
    debug(ROI_table.to_df())
    with pytest.raises(ValueError) as e:
        convert_ROI_table_to_indices(
            ROI_table,
            full_res_pxl_sizes_zyx=[1.0, 1.0, 1.0],
        )
    print(e.value)
    assert "negative array indices" in str(e.value)


def test_check_valid_ROI_indices():
    """
    Ref
    https://github.com/fractal-analytics-platform/fractal-tasks-core/issues/530.
    """
    # Indices starting at (0, 0, 0)
    list_indices = [
        [1, 2, 0, 1, 0, 1],
        [0, 1, 0, 1, 0, 1],
    ]
    check_valid_ROI_indices(list_indices, "FOV_ROI_table")
    check_valid_ROI_indices(list_indices, "something")

    # Indices that independently start at 0 on each axis
    list_indices = [
        [2, 3, 0, 1, 0, 1],
        [1, 2, 0, 1, 0, 1],
        [0, 1, 1, 2, 0, 1],
        [0, 1, 0, 1, 1, 2],
    ]
    check_valid_ROI_indices(list_indices, "something")
    check_valid_ROI_indices(list_indices, "FOV_ROI_table")

    # X indices do not start at 0
    list_indices = [
        [2, 3, 0, 1, 1, 2],
        [1, 2, 0, 1, 2, 3],
        [0, 1, 1, 2, 3, 5],
        [0, 1, 0, 1, 1, 2],
    ]
    check_valid_ROI_indices(list_indices, "something")
    with pytest.raises(ValueError) as e:
        check_valid_ROI_indices(list_indices, "FOV_ROI_table")
    print(str(e.value))
    assert "do not start with 0" in str(e.value)


def test_array_to_bounding_box_table_empty():
    """
    When trying to compute bounding boxes for a label array which has no labels
    (that is, it only has zeros), the output dataframe has zero rows (but it
    still has the `label` column, as expected).

    Also test `empty_bounding_box_table`.
    """
    mask_array = np.zeros((10, 100, 100))
    df = array_to_bounding_box_table(mask_array, pxl_sizes_zyx=[1.0, 1.0, 1.0])
    debug(df)
    assert df.shape[0] == 0
    assert "label" in df.columns

    df_empty = empty_bounding_box_table()
    pd.testing.assert_frame_equal(df, df_empty)


def test_array_to_bounding_box_table():
    """
    Test the new origin_zyx argument of array_to_bounding_box_table, ref
    https://github.com/fractal-analytics-platform/fractal-tasks-core/issues/460.
    """
    IMG_SIZE_X = 100
    IMG_SIZE_Y = 80
    PIXEL_SIZES = [0.4, 0.7, 1.2]
    masks1 = np.zeros((2, IMG_SIZE_Y, IMG_SIZE_X))
    masks1[:, 0:10, 0:12] = 1
    masks2 = np.zeros((2, IMG_SIZE_Y, IMG_SIZE_X))
    masks2[:, 0:10, 0:12] = 1
    masks2[:, 10:30, 10:50] = 2
    df1 = array_to_bounding_box_table(
        masks1,
        pxl_sizes_zyx=PIXEL_SIZES,
    )
    print(df1)
    print()
    assert df1.iloc[0].x_micrometer == 0.0
    assert df1.iloc[0].z_micrometer == 0.0
    df2 = array_to_bounding_box_table(
        masks2,
        pxl_sizes_zyx=PIXEL_SIZES,
        origin_zyx=(0, IMG_SIZE_Y, IMG_SIZE_X),
    )
    print(df2)
    print()
    assert df2.iloc[0].x_micrometer == IMG_SIZE_X * PIXEL_SIZES[-1]
    assert df2.iloc[0].y_micrometer == IMG_SIZE_Y * PIXEL_SIZES[-2]
    assert df2.iloc[0].len_y_micrometer == 10 * PIXEL_SIZES[-2]
    assert df2.iloc[0].len_x_micrometer == 12 * PIXEL_SIZES[-1]
    assert df2.iloc[1].len_y_micrometer == 20 * PIXEL_SIZES[-2]
    assert df2.iloc[1].len_x_micrometer == 40 * PIXEL_SIZES[-1]


# input shapes, regions, expected_output_shape
shapes = [
    (
        (10, 100, 100),
        (slice(0, 20), slice(0, 100), slice(0, 100)),
        (10, 100, 100),
    ),
    (
        (10, 100, 100),
        (slice(0, 5), slice(0, 100), slice(0, 100)),
        (5, 100, 100),
    ),
    (
        (10, 100, 100),
        (slice(0, 1), slice(0, 100), slice(0, 100)),
        (1, 100, 100),
    ),
    (
        (1, 100, 100),
        (slice(0, 20), slice(0, 100), slice(0, 100)),
        (1, 100, 100),
    ),
    (
        (1, 100, 100),
        (slice(0, 5), slice(0, 100), slice(0, 100)),
        (1, 100, 100),
    ),
    (
        (1, 100, 100),
        (slice(0, 1), slice(0, 100), slice(0, 100)),
        (1, 100, 100),
    ),
    ((100, 100), (slice(0, 20), slice(0, 100), slice(0, 100)), (1, 100, 100)),
    ((100, 100), (slice(0, 5), slice(0, 100), slice(0, 100)), (1, 100, 100)),
    ((100, 100), (slice(0, 1), slice(0, 100), slice(0, 100)), (1, 100, 100)),
]


@pytest.mark.parametrize("input_shape,region,expected_shape", shapes)
@pytest.mark.parametrize("compute", [True, False])
@pytest.mark.parametrize("return_as_3D", [True, False])
def test_load_region(
    input_shape, region, expected_shape, compute, return_as_3D
):
    da_array = da.ones(input_shape)
    output = load_region(
        da_array, region, compute=compute, return_as_3D=return_as_3D
    )
    expected_type = np.ndarray if compute else da.Array
    assert isinstance(output, expected_type)

    if return_as_3D is False and len(da_array.shape) == 2:
        expected_shape = expected_shape[1:]
    assert output.shape == expected_shape


def test_load_region_fail():
    with pytest.raises(ValueError) as e:
        load_region(
            data_zyx=da.ones((2, 3)),
            region=(slice(0, 1), slice(0, 1)),
        )
    debug(e.value)
    with pytest.raises(ValueError) as e:
        load_region(
            data_zyx=da.ones((2,)),
            region=(slice(0, 1), slice(0, 1), slice(0, 1)),
        )
    debug(e.value)
    with pytest.raises(ValueError) as e:
        load_region(
            data_zyx=da.ones((2, 3, 4, 5)),
            region=(slice(0, 1), slice(0, 1), slice(0, 1)),
        )
    debug(e.value)


EXPECTED_COLUMNS = [
    "x_micrometer",
    "y_micrometer",
    "z_micrometer",
    "len_x_micrometer",
    "len_y_micrometer",
    "len_z_micrometer",
]


def test_is_ROI_table_valid(tmp_path):
    # Write valid table to a zarr group
    columns = EXPECTED_COLUMNS.copy()
    adata = ad.AnnData(np.ones((1, len(columns))))
    adata.var_names = columns
    group = zarr.group(str(tmp_path / "group.zarr"))
    table_name = "table1"
    write_elem(group, table_name, adata)
    table_path = str(tmp_path / "group.zarr" / table_name)

    # Case 1: use_masks=False
    is_valid = is_ROI_table_valid(table_path=table_path, use_masks=False)
    assert is_valid is None

    # Case 2: use_masks=True, invalid attrs
    is_valid = is_ROI_table_valid(table_path=table_path, use_masks=True)
    assert not is_valid

    # Case 3: use_masks=True, valid attrs
    group[table_name].attrs["type"] = "masking_roi_table"
    group[table_name].attrs["instance_key"] = "label"
    group[table_name].attrs["region"] = {"path": "/tmp/"}
    is_valid = is_ROI_table_valid(table_path=table_path, use_masks=True)
    assert is_valid


def test_are_ROI_table_columns_valid():
    # Success
    columns = EXPECTED_COLUMNS.copy()
    adata = ad.AnnData(np.ones((1, len(columns))))
    adata.var_names = columns
    debug(adata)
    are_ROI_table_columns_valid(table=adata)
    # Failure
    columns = EXPECTED_COLUMNS.copy()
    columns[0] = "something_else"
    adata = ad.AnnData(np.ones((1, len(columns))))
    adata.var_names = columns
    debug(adata)
    with pytest.raises(ValueError):
        are_ROI_table_columns_valid(table=adata)


index_regions = [
    (
        [0, 1, 0, 540, 640, 1280],
        (slice(0, 1, None), slice(0, 540, None), slice(640, 1280, None)),
    ),
    (
        [0, 20, 100, 540, 900, 2000],
        (slice(0, 20, None), slice(100, 540, None), slice(900, 2000, None)),
    ),
]


@pytest.mark.parametrize("index,expected_results", index_regions)
def test_indices_to_region_conversion(index, expected_results):
    assert convert_indices_to_regions(index) == expected_results


def test_reset_origin():
    # Prepare ROI
    columns = EXPECTED_COLUMNS.copy()
    old_adata = ad.AnnData(np.ones((1, len(columns))))
    old_adata.var_names = columns
    debug(old_adata.X)
    # Reset origin
    new_adata = reset_origin(old_adata)
    debug(old_adata.X)
    debug(new_adata.X)
    # Check that new_adata was shifted
    assert abs(new_adata[:, "x_micrometer"].X[0, 0]) < 1e-10
    assert abs(new_adata[:, "y_micrometer"].X[0, 0]) < 1e-10
    assert abs(new_adata[:, "z_micrometer"].X[0, 0]) < 1e-10
    # Check that old_adata was not modified
    assert abs(old_adata[:, "x_micrometer"].X[0, 0] - 1.0) < 1e-10
    assert abs(old_adata[:, "y_micrometer"].X[0, 0] - 1.0) < 1e-10
    assert abs(old_adata[:, "z_micrometer"].X[0, 0] - 1.0) < 1e-10


def test_is_standard_roi_table():
    assert is_standard_roi_table("xxx_well_ROI_table_xxx")
    assert is_standard_roi_table("xxx_FOV_ROI_table_xxx")
    assert not is_standard_roi_table("something_else")


def test_search_first_ROI(testdata_path: Path):
    """
    See
    https://github.com/fractal-analytics-platform/fractal-tasks-core/issues/554
    """
    big_df = pd.read_csv(
        str(
            testdata_path
            / "site_metadata_ZebrafishMultiplexing_cycle0_new_rois.csv"
        )
    )
    full_res_pxl_sizes_zyx = [
        big_df["pixel_size_z"].iloc[0],
        big_df["pixel_size_y"].iloc[0],
        big_df["pixel_size_x"].iloc[0],
    ]

    well_ids = big_df.well_id.unique()
    for well_id in well_ids:
        debug(well_id)
        # Select a specific well from big_df
        df = big_df.loc[big_df["well_id"] == well_id, :].copy()
        # Construct and validate FOV ROIs
        FOV_ROI_table = prepare_FOV_ROI_table(df)
        list_indices = convert_ROI_table_to_indices(
            FOV_ROI_table,
            full_res_pxl_sizes_zyx=full_res_pxl_sizes_zyx,
        )
        check_valid_ROI_indices(list_indices, "FOV_ROI_table")
        # Construct and validate well ROI
        well_ROI_table = prepare_FOV_ROI_table(df)
        list_indices = convert_ROI_table_to_indices(
            well_ROI_table,
            full_res_pxl_sizes_zyx=full_res_pxl_sizes_zyx,
        )
        check_valid_ROI_indices(list_indices, "well_ROI_table")


def test_get_single_image_ROI():
    array_shape = (2, 3, 4)
    pixels_ZYX = (1.0, 0.5, 0.2)
    ROI = get_single_image_ROI(array_shape, pixels_ZYX)
    debug(ROI)
    debug(ROI.X)
    assert ROI.shape[0] == 1
    EXPECTED_DATA = np.array(
        (
            0,
            0,
            0,
            array_shape[2] * pixels_ZYX[2],
            array_shape[1] * pixels_ZYX[1],
            array_shape[0] * pixels_ZYX[0],
        )
    )
    assert np.allclose(EXPECTED_DATA, ROI.X)


def test_get_image_grid_ROIs():
    # CASE 1: all ROIs have the same size
    array_shape = (3, 4, 2)
    pixels_ZYX = (1.0, 0.5, 0.2)
    grid_shape_Y = 4
    grid_shape_X = 2
    ROI = get_image_grid_ROIs(
        array_shape,
        pixels_ZYX,
        grid_YX_shape=(grid_shape_Y, grid_shape_X),
    )
    debug(ROI.X)
    assert ROI.shape[0] == grid_shape_Y * grid_shape_X  # number of ROIS
    EXPECTED_DATA = np.array(
        (
            0,
            0,
            0,
            array_shape[2] * pixels_ZYX[2] / grid_shape_X,
            array_shape[1] * pixels_ZYX[1] / grid_shape_Y,
            array_shape[0] * pixels_ZYX[0],
        )
    )
    assert np.allclose(EXPECTED_DATA, ROI.X[0])
    assert np.allclose(pixels_ZYX[2], ROI.X[:, 3])  # X pixel size
    assert np.allclose(pixels_ZYX[1], ROI.X[:, 4])  # Y pixel size

    # CASE 2: non-commensurable division
    array_shape = (3, 10, 10)
    pixels_ZYX = (1.0, 0.5, 0.2)
    grid_shape_Y = 1
    grid_shape_X = 3
    ROI = get_image_grid_ROIs(
        array_shape,
        pixels_ZYX,
        grid_YX_shape=(grid_shape_Y, grid_shape_X),
    )
    debug(ROI.X)
    assert ROI.shape[0] == grid_shape_Y * grid_shape_X  # number of ROIS

    # Check that sum of len_x is equal to total X size
    assert np.allclose(ROI.X[:, 3].sum(), array_shape[2] * pixels_ZYX[2])
    # Check that last ROI ends at total X size
    assert np.allclose(
        ROI.X[-1, 3] + ROI.X[-1, 0], array_shape[2] * pixels_ZYX[2]
    )

    # Check data
    EXPECTED_DATA = np.array(
        [
            [0.0, 0.0, 0.0, 0.8, 5.0, 3.0],
            [0.8, 0.0, 0.0, 0.8, 5.0, 3.0],
            [1.6, 0.0, 0.0, 0.4, 5.0, 3.0],
        ]
    )
    assert np.allclose(EXPECTED_DATA, ROI.X)


def test_create_roi_table_from_empty_list():
    bbox_dataframe_list = []
    empty_roi_table = create_roi_table_from_df_list(bbox_dataframe_list)
    assert len(empty_roi_table) == 0


def test_create_roi_table_from_df_list():
    data1 = {
        "x_micrometer": [0.0, 26.0, 26.0],
        "y_micrometer": [0.0, 26.0, 13.0],
        "z_micrometer": [0.0, 0.0, 0.0],
        "len_x_micrometer": [104.0, 78.0, 104.0],
        "len_y_micrometer": [104.0, 78.0, 104.0],
        "len_z_micrometer": [2.0, 2.0, 2.0],
        "label": [1, 2, 3],
    }
    data2 = {
        "x_micrometer": [416.0, 442.0, 442.0],
        "y_micrometer": [0.0, 26.0, 13.0],
        "z_micrometer": [0.0, 0.0, 0.0],
        "len_x_micrometer": [104.0, 78.0, 104.0],
        "len_y_micrometer": [104.0, 78.0, 104.0],
        "len_z_micrometer": [2.0, 2.0, 2.0],
        "label": [4, 5, 6],
    }
    bbox_dataframe_list = [pd.DataFrame(data1), pd.DataFrame(data2)]
    roi_table = create_roi_table_from_df_list(bbox_dataframe_list)
    expected_rois = pd.DataFrame(
        [1, 2, 3, 4, 5, 6], columns=["label"], index=roi_table.obs.index
    )
    pd.testing.assert_frame_equal(roi_table.obs.astype(int), expected_rois)
    output_array = np.array(
        [
            [0.0, 0.0, 0.0, 104.0, 104.0, 2.0],
            [26.0, 26.0, 0.0, 78.0, 78.0, 2.0],
            [26.0, 13.0, 0.0, 104.0, 104.0, 2.0],
            [416.0, 0.0, 0.0, 104.0, 104.0, 2.0],
            [442.0, 26.0, 0.0, 78.0, 78.0, 2.0],
            [442.0, 13.0, 0.0, 104.0, 104.0, 2.0],
        ]
    )
    np.testing.assert_allclose(output_array, roi_table.X)


def test_create_roi_table_from_df_list_with_label_repeats():
    # Test that repeating labels are handled correctly
    data1 = {
        "x_micrometer": [0.0, 26.0, 26.0],
        "y_micrometer": [0.0, 26.0, 13.0],
        "z_micrometer": [0.0, 0.0, 0.0],
        "len_x_micrometer": [104.0, 78.0, 104.0],
        "len_y_micrometer": [104.0, 78.0, 104.0],
        "len_z_micrometer": [2.0, 2.0, 2.0],
        "label": [1, 2, 3],
    }
    data2 = {
        "x_micrometer": [416.0, 442.0, 442.0],
        "y_micrometer": [0.0, 26.0, 13.0],
        "z_micrometer": [0.0, 0.0, 0.0],
        "len_x_micrometer": [104.0, 78.0, 104.0],
        "len_y_micrometer": [104.0, 78.0, 104.0],
        "len_z_micrometer": [2.0, 2.0, 2.0],
        "label": [2, 3, 7],
    }
    bbox_dataframe_list = [pd.DataFrame(data1), pd.DataFrame(data2)]
    roi_table = create_roi_table_from_df_list(bbox_dataframe_list)
    expected_rois = pd.DataFrame(
        [1, 2, 3, 7], columns=["label"], index=roi_table.obs.index
    )
    pd.testing.assert_frame_equal(roi_table.obs.astype(int), expected_rois)
    output_array = np.array(
        [
            [0.0, 0.0, 0.0, 104.0, 104.0, 2.0],
            [26.0, 26.0, 0.0, 78.0, 78.0, 2.0],
            [26.0, 13.0, 0.0, 104.0, 104.0, 2.0],
            [442.0, 13.0, 0.0, 104.0, 104.0, 2.0],
        ]
    )
    np.testing.assert_allclose(output_array, roi_table.X)

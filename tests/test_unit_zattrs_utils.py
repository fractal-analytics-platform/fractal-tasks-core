import zarr
from devtools import debug

from fractal_tasks_core.utils import (
    _get_table_path_dict,
)
from fractal_tasks_core.utils import rescale_datasets


def test_rescale_datasets(tmp_path):
    """
    Test behavior of rescale_datasets in the presence of 3D or 4D scale
    transformations
    """

    # 3D scale transformation (ZYX)
    datasets = [
        dict(coordinateTransformations=[dict(type="scale", scale=[3, 2, 1])])
    ]
    new_datasets = rescale_datasets(
        datasets=datasets, coarsening_xy=2, reference_level=1
    )
    debug(new_datasets)
    assert new_datasets[0]["coordinateTransformations"][0]["scale"] == [
        3,
        4,
        2,
    ]

    # 4D scale transformation (CZYX)
    datasets = [
        dict(
            coordinateTransformations=[dict(type="scale", scale=[1, 3, 2, 1])]
        )
    ]
    new_datasets = rescale_datasets(
        datasets=datasets, coarsening_xy=2, reference_level=1
    )
    debug(new_datasets)
    assert new_datasets[0]["coordinateTransformations"][0]["scale"] == [
        1,
        3,
        4,
        2,
    ]


def test_get_table_path_dict(tmp_path):
    zarr_url = str(tmp_path / "plate.zarr/B/03/0")
    img_group = zarr.open_group(zarr_url)

    # Missing tables sub-group
    table_path_dict = _get_table_path_dict(zarr_url)
    debug(table_path_dict)
    assert table_path_dict == {}

    tables_group = img_group.create_group("tables")
    table_path_dict = _get_table_path_dict(zarr_url)
    debug(table_path_dict)
    assert table_path_dict == {}

    tables_group.attrs.update({"tables": ["table1", "table2"]})
    table_path_dict = _get_table_path_dict(zarr_url)
    debug(table_path_dict)
    assert table_path_dict.pop("table1") == str(f"{zarr_url}/tables/table1")
    assert table_path_dict.pop("table2") == str(f"{zarr_url}/tables/table2")

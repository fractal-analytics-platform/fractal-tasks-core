import json

import pytest
import zarr
from devtools import debug

from fractal_tasks_core.lib_zattrs_utils import extract_zyx_pixel_sizes
from fractal_tasks_core.lib_zattrs_utils import (
    get_acquisition_paths,
)
from fractal_tasks_core.lib_zattrs_utils import get_axes_names
from fractal_tasks_core.lib_zattrs_utils import (
    get_table_path_dict,
)
from fractal_tasks_core.lib_zattrs_utils import rescale_datasets  # noqa


def test_extract_zyx_pixel_sizes(tmp_path):
    """
    Test multiple invalid/valid calls to extract_zyx_pixel_sizes

    FIXME: transform into unit testing of ngff classes/methods
    """

    zattrs_path = tmp_path / ".zattrs"

    def _call_extract_zyx_pixel_sizes(_metadata):
        """
        Auxiliary function, to make the test more compact
        """
        with zattrs_path.open("w") as f:
            json.dump(metadata, f)
        return extract_zyx_pixel_sizes(zattrs_path=str(zattrs_path))

    # Case 1: fail for multiple multiscales
    metadata = dict(multiscales=[1, 2])
    with pytest.raises(ValueError) as e:
        _call_extract_zyx_pixel_sizes(metadata)
    debug(e.value)
    assert "There are 2 multiscales" in str(e.value)

    # Case 2: fail for global coordinateTransformations
    metadata = dict(multiscales=[dict(axes=[], coordinateTransformations=[])])
    with pytest.raises(NotImplementedError) as e:
        _call_extract_zyx_pixel_sizes(metadata)
    debug(e.value)
    assert "Global coordinateTransformations" in str(e.value)

    # Case 3: fail for missing scale transformation
    metadata = dict(
        multiscales=[
            dict(axes=[], datasets=[dict(coordinateTransformations=[])])
        ]
    )
    with pytest.raises(ValueError) as e:
        _call_extract_zyx_pixel_sizes(metadata)
    debug(e.value)
    assert "No scale transformation found" in str(e.value)

    # Case 4: success, with 4D (CZXY) scale transformation
    metadata = dict(
        multiscales=[
            dict(
                axes=[],
                datasets=[
                    dict(
                        coordinateTransformations=[
                            dict(type="scale", scale=[1, 2, 2, 2])
                        ]
                    )
                ],
            )
        ]
    )
    out = _call_extract_zyx_pixel_sizes(metadata)
    debug(out)
    assert out == [2, 2, 2]

    # Case 5: success, with 3D (ZYX) scale transformation
    metadata = dict(
        multiscales=[
            dict(
                axes=[],
                datasets=[
                    dict(
                        coordinateTransformations=[
                            dict(type="scale", scale=[2, 2, 2])
                        ]
                    )
                ],
            )
        ]
    )
    out = _call_extract_zyx_pixel_sizes(metadata)
    debug(out)
    assert out == [2, 2, 2]

    # Case 6: fail because pixel sizes are too small
    metadata = dict(
        multiscales=[
            dict(
                axes=[],
                datasets=[
                    dict(
                        coordinateTransformations=[
                            dict(type="scale", scale=[2, 2, 1e-20])
                        ]
                    )
                ],
            )
        ]
    )
    with pytest.raises(ValueError) as e:
        _call_extract_zyx_pixel_sizes(metadata)
    debug(e.value)


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


def test_get_acquisition_paths():

    # FIXME: transform into unit testing of ngff classes/methods

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


def test_get_table_path_dict(tmp_path):

    input_path = tmp_path
    component = "plate.zarr/B/03/0"
    img_group = zarr.open_group(str(input_path / component))

    # Missing tables sub-group
    table_path_dict = get_table_path_dict(input_path, component)
    debug(table_path_dict)
    assert table_path_dict == {}

    tables_group = img_group.create_group("tables")
    table_path_dict = get_table_path_dict(input_path, component)
    debug(table_path_dict)
    assert table_path_dict == {}

    tables_group.attrs.update({"tables": ["table1", "table2"]})
    table_path_dict = get_table_path_dict(input_path, component)
    debug(table_path_dict)
    assert table_path_dict.pop("table1") == str(
        input_path / component / "tables/table1"
    )
    assert table_path_dict.pop("table2") == str(
        input_path / component / "tables/table2"
    )


def test_get_axes_names():
    attrs = {
        "multiscales": [
            {
                "axes": [
                    {"name": "c", "type": "channel"},
                    {"name": "z", "type": "space", "unit": "micrometer"},
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"},
                ]
            }
        ]
    }
    target_axes = ["c", "z", "y", "x"]
    axes = get_axes_names(attrs)

    assert target_axes == axes


def test_get_axes_names_errors():
    attrs_list = [
        {},
        {"multiscales": "Test"},
        {"multiscales": [{"Test": "test1"}]},
    ]
    for attrs in attrs_list:
        with pytest.raises(ValueError):
            get_axes_names(attrs)

import anndata as ad
import numpy as np
import pytest
import zarr
from devtools import debug

from fractal_tasks_core.lib_write import _write_elem_with_overwrite
from fractal_tasks_core.lib_write import open_zarr_group_with_overwrite
from fractal_tasks_core.lib_write import OverwriteNotAllowedError
from fractal_tasks_core.lib_write import prepare_label_group


def test_open_zarr_group_with_overwrite(tmp_path, caplog):
    """
    Test open_zarr_group_with_overwrite

    See
    https://github.com/fractal-analytics-platform/fractal-tasks-core/issues/485
    """

    path_a = str(tmp_path / "group_a.zarr")
    path_b = str(tmp_path / "group_b.zarr")
    path_c = str(tmp_path / "group_c.zarr")

    # If `overwrite=False` and the group does not exist, create it
    group_a = open_zarr_group_with_overwrite(path_a, overwrite=False)
    group_a.create_group("a1")
    group_a.create_group("a2")
    debug(group_a.info)
    assert set(group_a.group_keys()) == {"a1", "a2"}
    print()

    # If `overwrite=True` and the group does not exist, create it
    group_b = open_zarr_group_with_overwrite(path_b, overwrite=True)
    group_b.create_group("b1")
    group_b.create_group("b2")
    debug(group_b.info)
    assert set(group_b.group_keys()) == {"b1", "b2"}
    print()

    # If `overwrite=True` and the group already exists, replace it with a new
    # empty group
    group_a_new = open_zarr_group_with_overwrite(path_a, overwrite=True)
    debug(group_a_new.info)
    assert set(group_a_new.group_keys()) == set()
    print()

    # If `overwrite=False` and the group already exists, fail with a relevant
    # error
    with pytest.raises(OverwriteNotAllowedError) as e:
        open_zarr_group_with_overwrite(path_b, overwrite=False)
    print(e.value)

    # If `mode` is also included, a warning is raised but there is no error
    caplog.clear()
    open_zarr_group_with_overwrite(path_c, overwrite=False, mode="something")
    debug(caplog.text)
    assert "Overriding mode='something' with new_mode" in caplog.text


def test_write_elem_with_overwrite(tmp_path):
    """
    Test wrapper of write_elem anndata function.
    """

    # Generate fake ROI tables
    ROI_table_1 = ad.AnnData(np.ones((1, 1)))
    ROI_table_2 = ad.AnnData(np.ones((2, 2)))
    ROI_table_3 = ad.AnnData(np.ones((3, 3)))
    ROI_table_4 = ad.AnnData(np.ones((4, 4)))

    # Create zarr groups for image and tables
    zarr_path = str(tmp_path / "my_image.zarr")
    tables_group = zarr.open(zarr_path, mode="w", path="tables")
    debug(set(tables_group.group_keys()))
    assert set(tables_group.group_keys()) == set()

    # If overwrite=True and the sub-group does not exist, create it
    _write_elem_with_overwrite(
        tables_group, "table_a", ROI_table_1, overwrite=True
    )
    debug(set(tables_group.group_keys()))
    assert set(tables_group.group_keys()) == {"table_a"}

    # If overwrite=False and the sub-group does not exist, create it
    _write_elem_with_overwrite(
        tables_group, "table_b", ROI_table_2, overwrite=False
    )
    debug(set(tables_group.group_keys()))
    assert set(tables_group.group_keys()) == {"table_a", "table_b"}

    # If overwrite=True and the sub-group already exists, replace it
    _write_elem_with_overwrite(
        tables_group, "table_a", ROI_table_3, overwrite=True
    )
    debug(set(tables_group.group_keys()))
    assert set(tables_group.group_keys()) == {"table_a", "table_b"}
    subgroup = tables_group["table_a"]
    assert subgroup.X.shape == (3, 3)  # Verify that it was overwritten

    # If overwrite=False and the sub-group already exists, fail
    with pytest.raises(OverwriteNotAllowedError):
        _write_elem_with_overwrite(
            tables_group, "table_a", ROI_table_4, overwrite=False
        )
    assert subgroup.X.shape == (3, 3)  # Verify that it was not overwritten


def test_prepare_label_group(tmp_path):
    """
    Test some specific behaviors of `prepare_label_group`.
    """

    # Create zarr groups for image and labels
    zarr_path = str(tmp_path / "my_image.zarr")
    image_group = zarr.open(zarr_path, mode="w")

    # Run prepare_label_group
    prepare_label_group(image_group, "label_a")
    assert set(image_group.group_keys()) == {"labels"}
    assert image_group["labels"].attrs.asdict() == dict(labels=["label_a"])

    # Run prepare_label_group again, with overwrite=True
    prepare_label_group(image_group, "label_a", overwrite=True)
    assert set(image_group.group_keys()) == {"labels"}
    assert image_group["labels"].attrs.asdict() == dict(labels=["label_a"])

    # Run prepare_label_group, with label_attrs parameters
    KEY = "KEY"
    VALUE = "VALUE"
    label_b_group = prepare_label_group(
        image_group, "label_b", label_attrs=dict(KEY=VALUE)
    )
    assert image_group["labels"].attrs.asdict() == dict(
        labels=["label_a", "label_b"]
    )
    assert label_b_group.attrs[KEY] == VALUE

    # Verify the overwrite=False failure if sub-group already exists
    image_group["labels"].create_group("label_c")
    with pytest.raises(OverwriteNotAllowedError) as e:
        prepare_label_group(image_group, "label_c")
    assert str(e.value).startswith("Sub-group ")

    # Verify the overwrite=False failure if item already exists in labels
    # attribute
    image_group["labels"].attrs["labels"] = ["label_a", "label_b", "label_d"]
    with pytest.raises(OverwriteNotAllowedError) as e:
        prepare_label_group(image_group, "label_d")
    assert str(e.value).startswith("Item ")

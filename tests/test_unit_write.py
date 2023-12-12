import logging

import anndata as ad
import numpy as np
import pytest
import zarr
from devtools import debug

from fractal_tasks_core.labels import prepare_label_group
from fractal_tasks_core.tables.v1 import _write_elem_with_overwrite
from fractal_tasks_core.zarr import open_zarr_group_with_overwrite
from fractal_tasks_core.zarr import OverwriteNotAllowedError


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


def test_prepare_label_group(tmp_path, caplog):
    """
    Test some specific behaviors of `prepare_label_group`.
    """

    # Prepare label attributes with name OLD_NAME
    ATTRS = {
        "image-label": {
            "version": "0.4",
            "source": dict(image="../../"),
        },
        "multiscales": [
            {
                "name": "OLD_NAME",
                "version": "0.4",
                "axes": [
                    {
                        "name": "z",
                        "type": "space",
                        "unit": "micrometer",
                    },
                    {
                        "name": "y",
                        "type": "space",
                        "unit": "micrometer",
                    },
                    {
                        "name": "x",
                        "type": "space",
                        "unit": "micrometer",
                    },
                ],
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {
                                "type": "scale",
                                "scale": [
                                    1.0,
                                    1.3,
                                    1.3,
                                ],
                            },
                        ],
                    },
                ],
            },
        ],
    }

    # Create zarr groups for image and labels
    zarr_path = str(tmp_path / "my_image.zarr")
    image_group = zarr.open(zarr_path, mode="w")

    # Run prepare_label_group
    label_name = "label_A"
    caplog.clear()
    caplog.set_level(logging.WARNING)
    prepare_label_group(image_group, label_name, label_attrs=ATTRS)
    assert set(image_group.group_keys()) == {"labels"}
    assert image_group["labels"].attrs.asdict() == dict(labels=[label_name])
    # Check that name was overwritten in Zarr attributes
    label_group = zarr.open(f"{zarr_path}/labels/{label_name}", mode="r")
    assert label_group.attrs["multiscales"][0]["name"] == label_name
    assert "Setting multiscale name to" in caplog.text

    # Run prepare_label_group again, with overwrite=True
    prepare_label_group(
        image_group, label_name, overwrite=True, label_attrs=ATTRS
    )
    assert set(image_group.group_keys()) == {"labels"}
    assert image_group["labels"].attrs.asdict() == dict(labels=[label_name])
    label_group = zarr.open(f"{zarr_path}/labels/{label_name}", mode="r")
    assert label_group.attrs["multiscales"][0]["name"] == label_name

    # Verify the overwrite=False failure if sub-group already exists
    label_name = "label_B"
    image_group["labels"].create_group(label_name)
    with pytest.raises(OverwriteNotAllowedError) as e:
        prepare_label_group(image_group, label_name, label_attrs=ATTRS)
    assert str(e.value).startswith("Sub-group ")

    # Verify the overwrite=False failure if item already exists in labels
    # attribute
    label_name = "label_C"
    image_group["labels"].attrs["labels"] = ["something", label_name]
    with pytest.raises(OverwriteNotAllowedError) as e:
        prepare_label_group(image_group, label_name, label_attrs=ATTRS)
    assert str(e.value).startswith("Item ")

    # Verify failures in case of missing or invalid label-group attributes
    label_name = "label_D"
    with pytest.raises(TypeError):
        prepare_label_group(image_group, label_name)
    with pytest.raises(ValueError) as e:
        prepare_label_group(
            image_group, label_name, label_attrs={"something": "invalid"}
        )
    assert "do not comply" in str(e.value)

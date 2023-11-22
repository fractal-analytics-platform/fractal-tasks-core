from typing import Any
from typing import Optional

import anndata as ad
import numpy as np
import pytest
import zarr
from devtools import debug

from fractal_tasks_core import __FRACTAL_TABLE_VERSION__
from fractal_tasks_core.lib_tables import write_table
from fractal_tasks_core.lib_write import OverwriteNotAllowedError


def test_write_table(tmp_path):
    """
    Test some specific behaviors of `write_table`, especially the logic which
    is not part of `_write_elem_with_overwrite`.
    """

    # Generate fake ROI tables
    ROI_table_1 = ad.AnnData(np.ones((1, 1)))
    ROI_table_2 = ad.AnnData(np.ones((2, 2)))

    # Create zarr groups for image and tables
    zarr_path = str(tmp_path / "my_image.zarr")
    image_group = zarr.open(zarr_path, mode="w")

    # Run write_table
    table_a_group = write_table(image_group, "table_a", ROI_table_1)
    assert set(image_group.group_keys()) == {"tables"}
    assert image_group["tables"].attrs.asdict() == dict(tables=["table_a"])
    for key in ["region", "instance_key", "type"]:
        assert key not in table_a_group.attrs.keys()
    assert (
        table_a_group.attrs["fractal_table_version"]
        == __FRACTAL_TABLE_VERSION__
    )

    # Run write_table again, with overwrite=True
    table_a_group = write_table(
        image_group, "table_a", ROI_table_2, overwrite=True
    )
    assert set(image_group.group_keys()) == {"tables"}
    assert image_group["tables"].attrs.asdict() == dict(tables=["table_a"])
    for key in ["region", "instance_key", "type"]:
        assert key not in table_a_group.attrs.keys()
    assert table_a_group.X.shape == (2, 2)  # Verify that it was overwritten

    # Run write_table, with table_attrs parameters
    KEY = "KEY"
    VALUE = "VALUE"
    table_b_group = write_table(
        image_group, "table_b", ROI_table_2, table_attrs=dict(KEY=VALUE)
    )
    assert image_group["tables"].attrs.asdict() == dict(
        tables=["table_a", "table_b"]
    )
    debug(table_b_group.attrs.asdict())
    assert table_b_group.attrs[KEY] == VALUE

    # Verify the overwrite=False failure if sub-group already exists
    image_group["tables"].create_group("table_c")
    with pytest.raises(OverwriteNotAllowedError) as e:
        write_table(image_group, "table_c", ROI_table_2)
    assert str(e.value).startswith("Sub-group ")

    # Verify the overwrite=False failure if item already exists in tables
    # attribute
    image_group["tables"].attrs["tables"] = ["table_a", "table_b", "table_d"]
    with pytest.raises(OverwriteNotAllowedError) as e:
        write_table(image_group, "table_d", ROI_table_2)
    assert str(e.value).startswith("Item ")


def test_write_table_warnings(tmp_path, caplog):
    """
    Test that the appropriate warnings are raised when not complying with the
    new proposed table specs.
    """

    table = ad.AnnData(np.ones((2, 2)))
    zarr_path = str(tmp_path / "my_image.zarr")
    img_group = zarr.open(zarr_path, mode="w")

    def _check_warnings(
        _ATTRS: dict[str, Any],
        expect_warning: bool = True,
        warning_message_contains: Optional[str] = None,
    ):
        caplog.clear()
        write_table(
            img_group, "table", table, table_attrs=_ATTRS, overwrite=True
        )
        debug(caplog.text)
        if warning_message_contains is None:
            WARNING_MSG = "does not comply with "
        else:
            WARNING_MSG = warning_message_contains
        if expect_warning:
            assert WARNING_MSG in caplog.text
        else:
            assert WARNING_MSG not in caplog.text

    # Run without warnings
    ATTRS = dict(
        type="masking_roi_table",
        region=dict(path="../labels/something"),
        instance_key="label",
    )
    _check_warnings(ATTRS, expect_warning=False)

    # Run with warnings, case 1
    ATTRS = dict(
        type="masking_roi_table",
        region=dict(path="../labels/something"),
    )
    _check_warnings(ATTRS)

    # Run with warnings, case 2
    ATTRS = dict(
        type="masking_roi_table",
        instance_key="label",
        region=dict(key="value"),
    )
    _check_warnings(ATTRS)

    # Run with warnings, case 3
    ATTRS = dict(
        type="masking_roi_table",
        instance_key="label",
    )
    _check_warnings(ATTRS)

    # Run with warnings, case 4
    ATTRS = dict(
        type="INVALID_TABLE_TYPE",
        instance_key="label",
    )
    _check_warnings(ATTRS, warning_message_contains="Unknown table type")

import anndata as ad
import numpy as np
import pytest
import zarr
from devtools import debug

from fractal_tasks_core import __FRACTAL_TABLE_VERSION__
from fractal_tasks_core.tables import write_table
from fractal_tasks_core.tables.v1 import FeatureTableAttrs
from fractal_tasks_core.tables.v1 import MaskingROITableAttrs
from fractal_tasks_core.zarr import OverwriteNotAllowedError


TYPE = "some-arbitrary-type"


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
    table_a_group = write_table(
        image_group, "table_a", ROI_table_1, table_type=TYPE
    )
    assert set(image_group.group_keys()) == {"tables"}
    assert image_group["tables"].attrs.asdict() == dict(tables=["table_a"])
    assert "region" not in table_a_group.attrs.keys()
    assert "instance_key" not in table_a_group.attrs.keys()
    assert table_a_group.attrs["type"] == TYPE
    assert (
        table_a_group.attrs["fractal_table_version"]
        == __FRACTAL_TABLE_VERSION__
    )  # noqa

    # Run write_table again, with overwrite=True
    table_a_group = write_table(
        image_group, "table_a", ROI_table_2, overwrite=True, table_type=TYPE
    )
    assert set(image_group.group_keys()) == {"tables"}
    assert image_group["tables"].attrs.asdict() == dict(tables=["table_a"])
    assert "region" not in table_a_group.attrs.keys()
    assert "instance_key" not in table_a_group.attrs.keys()
    assert table_a_group.attrs["type"] == TYPE
    assert table_a_group.X.shape == (2, 2)  # Verify that it was overwritten

    # Run write_table, with both table_type and table_attrs parameters
    table_b_group = write_table(
        image_group,
        "table_b",
        ROI_table_2,
        table_type=TYPE,
        table_attrs={"type": "wrong"},
    )
    assert image_group["tables"].attrs.asdict() == dict(
        tables=["table_a", "table_b"]
    )
    assert table_b_group.attrs["type"] == TYPE

    # Run write_table, without specifying type
    with pytest.raises(ValueError) as e:
        table_b_group = write_table(
            image_group,
            "table_b",
            ROI_table_2,
            overwrite=True,
            table_attrs={"something": "else"},
        )
    assert "Missing attribute `type`" in str(e.value)

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


def test_write_table_validation_errors(tmp_path, caplog):
    """
    Test that the appropriate errors are raised when not complying with the
    table specs.
    """

    table = ad.AnnData(np.ones((2, 2)))
    zarr_path = str(tmp_path / "my_image.zarr")
    img_group = zarr.open(zarr_path, mode="w")

    # Valid custom table
    ATTRS = dict(
        type="some_custom_table",
        something="else",
    )
    write_table(img_group, "table", table, overwrite=True, table_attrs=ATTRS)

    # Valid roi_table
    ATTRS = dict(
        type="roi_table",
        something="else",
    )
    write_table(img_group, "table", table, overwrite=True, table_attrs=ATTRS)

    # Valid masking_roi_table
    ATTRS = dict(
        type="masking_roi_table",
        region=dict(path="../labels/something"),
        instance_key="label",
    )
    write_table(img_group, "table", table, overwrite=True, table_attrs=ATTRS)

    # Invalid masking_roi_table
    ATTRS = dict(
        type="masking_roi_table",
        region=dict(path="../labels/something"),
    )
    with pytest.raises(ValueError):
        write_table(
            img_group, "table", table, overwrite=True, table_attrs=ATTRS
        )

    # Valid feature_table
    ATTRS = dict(
        type="feature_table",
        region=dict(path="../labels/something"),
        instance_key="label",
    )
    write_table(img_group, "table", table, overwrite=True, table_attrs=ATTRS)

    # Invalid feature_table
    ATTRS = dict(
        type="feature_table",
        region=dict(path="../labels/something"),
    )
    with pytest.raises(ValueError):
        write_table(
            img_group, "table", table, overwrite=True, table_attrs=ATTRS
        )


def test_write_table_V2_not_implemented(tmp_path):

    zarr_path = str(tmp_path / "my_image.zarr")
    img_group = zarr.open(zarr_path, mode="w")
    table = ad.AnnData(np.ones((2, 2)))

    with pytest.raises(NotImplementedError) as e:
        write_table(
            img_group,
            "table",
            table,
            table_attrs={"fractal_table_version": "2"},
        )
    debug(e.value)
    assert "not supported" in str(e.value)


@pytest.mark.filterwarnings("error::FutureWarning")
def test_MaskingROITableAttrs():
    # Valid instance
    MaskingROITableAttrs(
        type="masking_roi_table",
        region={"path": "../labels/something"},
        instance_key="label",
    )
    # FutureWarning (transformed into error, in this test)
    with pytest.raises(FutureWarning):
        MaskingROITableAttrs(
            type="ngff:region_table",
            region={"path": "../labels/something"},
            instance_key="label",
        )


@pytest.mark.filterwarnings("error::FutureWarning")
def test_FeatureTableAttrs():
    # Valid instance
    FeatureTableAttrs(
        type="feature_table",
        region={"path": "../labels/something"},
        instance_key="label",
    )
    # FutureWarning (transformed into error, in this test)
    with pytest.raises(FutureWarning):
        FeatureTableAttrs(
            type="ngff:region_table",
            region={"path": "../labels/something"},
            instance_key="label",
        )

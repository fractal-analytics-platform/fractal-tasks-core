from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import zarr as za
from ngio import create_empty_ome_zarr, open_ome_zarr_container

from fractal_tasks_core._threshold_segmentation_utils import (
    InputChannel,
    ThresholdConfiguration,
)
from fractal_tasks_core.measure_features import (
    IntensityFeatures,
    ShapeFeatures,
    join_tables,
    measure_features,
    region_props_features_func,
)
from fractal_tasks_core.threshold_segmentation import threshold_segmentation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_roi(name: str = "roi_0") -> Mock:
    roi = Mock()
    roi.get_name.return_value = name
    return roi


def _make_zarr_with_label(
    tmp_path: Path,
    shape: tuple = (1, 4, 32, 32),
    axes: str = "czyx",
    label_name: str = "nuclei",
) -> Path:
    """Create an OME-Zarr with image data and a label produced by threshold_segmentation."""
    store = tmp_path / "image.zarr"
    kw: dict = dict(
        store=store, shape=shape, pixelsize=0.1, overwrite=False, axes_names=axes
    )
    if "z" in axes:
        kw["z_spacing"] = 0.5
    create_empty_ome_zarr(**kw)

    root = za.open(str(store), mode="r+")
    data = np.zeros(shape, dtype=np.uint16)
    if len(shape) == 4:  # CZYX
        data[:, :, 8:20, 8:20] = 500
    else:  # CYX
        data[:, 8:20, 8:20] = 500
    root["0"][:] = data

    threshold_segmentation(
        zarr_url=str(store),
        channels=InputChannel(mode="index", identifier="0"),
        label_name=label_name,
        method=ThresholdConfiguration(threshold=100),
        overwrite=True,
    )
    return store


# ---------------------------------------------------------------------------
# Unit tests — ShapeFeatures / IntensityFeatures
# ---------------------------------------------------------------------------


def test_shape_features_property_names() -> None:
    names = ShapeFeatures().property_names
    assert set(names) == {
        "area",
        "area_bbox",
        "axis_major_length",
        "axis_minor_length",
        "solidity",
    }


def test_intensity_features_property_names() -> None:
    names = IntensityFeatures().property_names
    assert set(names) == {
        "mean_intensity",
        "max_intensity",
        "min_intensity",
        "std_intensity",
    }


# ---------------------------------------------------------------------------
# Unit tests — join_tables
# ---------------------------------------------------------------------------


def test_join_tables_single() -> None:
    table = {"label": [1, 2], "area": [100.0, 200.0]}
    df = join_tables([table])
    assert list(df.index) == [1, 2]
    assert "area" in df.columns


def test_join_tables_multiple() -> None:
    t1 = {"label": [1], "area": [100.0]}
    t2 = {"label": [2], "area": [200.0]}
    df = join_tables([t1, t2])
    assert len(df) == 2
    assert set(df.index) == {1, 2}


def test_join_tables_empty_raises() -> None:
    with pytest.raises(AssertionError):
        join_tables([])


# ---------------------------------------------------------------------------
# Unit tests — region_props_features_func
# ---------------------------------------------------------------------------


def test_region_props_features_func_2d_shape_features() -> None:
    """YXC input (C=1) with ShapeFeatures: columns and region name are correct."""
    image = np.zeros((32, 32, 1), dtype=np.float32)
    image[10:15, 10:15, 0] = 100.0
    label_arr = np.zeros((32, 32, 1), dtype=np.int32)
    label_arr[10:15, 10:15, 0] = 1

    result = region_props_features_func(
        image=image,
        label=label_arr,
        roi=_mock_roi("roi_0"),
        list_features=[ShapeFeatures()],
    )
    assert "area" in result
    assert result["region"] == ["roi_0"]
    assert len(result["label"]) == 1


def test_region_props_features_func_intensity_features() -> None:
    """YXC input: IntensityFeatures columns are present in result."""
    image = np.zeros((32, 32, 1), dtype=np.float32)
    image[10:15, 10:15, 0] = 50.0
    label_arr = np.zeros((32, 32, 1), dtype=np.int32)
    label_arr[10:15, 10:15, 0] = 1

    result = region_props_features_func(
        image=image,
        label=label_arr,
        roi=_mock_roi(),
        list_features=[IntensityFeatures()],
    )
    # skimage appends a channel suffix ("-0") when intensity_image has a channel dim
    assert any(k.startswith("mean_intensity") for k in result)


def test_region_props_features_func_wrong_ndim_raises() -> None:
    """Image without channel dim (ndim=2) raises AssertionError."""
    image = np.zeros((32, 32), dtype=np.float32)
    label_arr = np.zeros((32, 32), dtype=np.int32)
    with pytest.raises(AssertionError):
        region_props_features_func(
            image=image,
            label=label_arr,
            roi=_mock_roi(),
            list_features=[ShapeFeatures()],
        )


def test_region_props_features_func_multichannel_label_raises() -> None:
    """Label with C > 1 raises AssertionError."""
    image = np.zeros((32, 32, 2), dtype=np.float32)
    label_arr = np.zeros((32, 32, 2), dtype=np.int32)
    with pytest.raises(AssertionError):
        region_props_features_func(
            image=image,
            label=label_arr,
            roi=_mock_roi(),
            list_features=[ShapeFeatures()],
        )


# ---------------------------------------------------------------------------
# Integration tests — measure_features
# ---------------------------------------------------------------------------


def test_measure_features_basic(tmp_path: Path) -> None:
    """Feature table is created under the default name."""
    store = _make_zarr_with_label(tmp_path)

    measure_features(
        zarr_url=str(store),
        label_image_name="nuclei",
        features=[ShapeFeatures()],
    )

    ome = open_ome_zarr_container(str(store))
    assert "region_props_features" in ome.list_tables()


def test_measure_features_shape_and_intensity(tmp_path: Path) -> None:
    """Combining ShapeFeatures and IntensityFeatures succeeds."""
    store = _make_zarr_with_label(tmp_path)

    measure_features(
        zarr_url=str(store),
        label_image_name="nuclei",
        features=[ShapeFeatures(), IntensityFeatures()],
    )

    ome = open_ome_zarr_container(str(store))
    assert "region_props_features" in ome.list_tables()


def test_measure_features_2d(tmp_path: Path) -> None:
    """Feature extraction works on a 2D (CYX) image."""
    store = _make_zarr_with_label(tmp_path, shape=(1, 32, 32), axes="cyx")

    measure_features(
        zarr_url=str(store),
        label_image_name="nuclei",
        features=[ShapeFeatures()],
    )

    ome = open_ome_zarr_container(str(store))
    assert "region_props_features" in ome.list_tables()


def test_measure_features_overwrite_true(tmp_path: Path) -> None:
    """Running twice with overwrite=True succeeds."""
    store = _make_zarr_with_label(tmp_path)
    kwargs = dict(
        zarr_url=str(store),
        label_image_name="nuclei",
        features=[ShapeFeatures()],
        overwrite=True,
    )
    measure_features(**kwargs)
    measure_features(**kwargs)


def test_measure_features_overwrite_false_raises(tmp_path: Path) -> None:
    """Running a second time with overwrite=False raises FileExistsError."""
    store = _make_zarr_with_label(tmp_path)

    measure_features(
        zarr_url=str(store),
        label_image_name="nuclei",
        features=[ShapeFeatures()],
        overwrite=True,
    )

    with pytest.raises(FileExistsError):
        measure_features(
            zarr_url=str(store),
            label_image_name="nuclei",
            features=[ShapeFeatures()],
            overwrite=False,
        )


def test_measure_features_custom_table_name(tmp_path: Path) -> None:
    """output_table_name is stored under the given name."""
    store = _make_zarr_with_label(tmp_path)

    measure_features(
        zarr_url=str(store),
        label_image_name="nuclei",
        output_table_name="my_features",
        features=[ShapeFeatures()],
    )

    ome = open_ome_zarr_container(str(store))
    assert "my_features" in ome.list_tables()

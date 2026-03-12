from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from ngio import create_empty_ome_zarr, open_ome_zarr_container

from fractal_tasks_core._threshold_segmentation_utils import (
    InputChannel,
    ThresholdConfiguration,
)
from fractal_tasks_core.measure_features import (
    IntensityFeatures,
    ShapeFeatures,
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
    """Create an OME-Zarr with a label produced by threshold_segmentation."""
    store = tmp_path / "image.zarr"
    kw: dict = dict(
        store=store, shape=shape, pixelsize=0.1, overwrite=False, axes_names=axes
    )
    if "z" in axes:
        kw["z_spacing"] = 0.5
    ome_zarr = create_empty_ome_zarr(**kw)

    image = ome_zarr.get_image()
    data = np.zeros(shape, dtype=np.uint16)
    if len(shape) == 4:  # CZYX
        data[:, :, 8:20, 8:20] = 500
    else:  # CYX
        data[:, 8:20, 8:20] = 500
    image.set_array(data)
    image.consolidate()

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
    names = ShapeFeatures().property_names(is_2d=True)
    assert set(names) == {
        "area",
        "area_bbox",
        "num_pixels",
        "equivalent_diameter_area",
        "axis_major_length",
        "axis_minor_length",
        "euler_number",
        "feret_diameter_max",
        "perimeter",
        "perimeter_crofton",
        "eccentricity",
        "orientation",
    }
    names = ShapeFeatures().property_names(is_2d=False)
    assert set(names) == {
        "area",
        "area_bbox",
        "num_pixels",
        "equivalent_diameter_area",
        "axis_major_length",
        "axis_minor_length",
        "euler_number",
    }
    names = ShapeFeatures(include_convex_hull_properties=True).property_names(
        is_2d=True
    )
    assert set(names) == {
        "area",
        "area_bbox",
        "num_pixels",
        "equivalent_diameter_area",
        "axis_major_length",
        "axis_minor_length",
        "euler_number",
        "feret_diameter_max",
        "perimeter",
        "perimeter_crofton",
        "eccentricity",
        "orientation",
        "area_convex",
        "area_filled",
        "extent",
        "solidity",
    }


def test_intensity_features_property_names() -> None:
    names = IntensityFeatures().property_names(is_2d=True)
    assert set(names) == {
        "intensity_mean",
        "intensity_median",
        "intensity_max",
        "intensity_min",
        "intensity_std",
    }


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
        properties=["label"] + ShapeFeatures().property_names(is_2d=True),
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
        properties=["label"] + IntensityFeatures().property_names(is_2d=True),
    )
    # skimage appends a channel suffix ("-0") when intensity_image has a channel dim
    assert any(k.startswith("intensity_mean") for k in result)


def test_region_props_features_func_wrong_ndim_raises() -> None:
    """Image without channel dim (ndim=2) raises AssertionError."""
    image = np.zeros((32, 32), dtype=np.float32)
    label_arr = np.zeros((32, 32), dtype=np.int32)
    with pytest.raises(AssertionError):
        region_props_features_func(
            image=image,
            label=label_arr,
            roi=_mock_roi(),
            properties=ShapeFeatures().property_names(is_2d=True),
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
            properties=ShapeFeatures().property_names(is_2d=True),
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
    measure_features(
        zarr_url=str(store),
        label_image_name="nuclei",
        features=[ShapeFeatures()],
        overwrite=True,
    )
    measure_features(
        zarr_url=str(store),
        label_image_name="nuclei",
        features=[ShapeFeatures()],
        overwrite=True,
    )


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

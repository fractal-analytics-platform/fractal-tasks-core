from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from ngio import create_empty_ome_zarr, open_ome_zarr_container

from fractal_tasks_core._measure_features_utils import (
    AdvancedOptions,
    IntensityFeatures,
    ShapeFeatures,
    region_props_features_func,
)
from fractal_tasks_core._threshold_segmentation_utils import (
    InputChannel,
    SimpleThresholdConfiguration,
)
from fractal_tasks_core.measure_features import measure_features
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
    kw: dict = {
        "store": store,
        "shape": shape,
        "pixelsize": 0.1,
        "overwrite": False,
        "axes_names": axes,
    }
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
        channel=InputChannel(mode="index", identifier="0"),
        output_label_name=label_name,
        segmentation_method=SimpleThresholdConfiguration(threshold=100),
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


def test_intensity_features_to_channel_selection_models() -> None:
    """Empty list and None both return None; a populated list returns models."""
    from fractal_tasks_core._measure_features_utils import InputChannel

    assert IntensityFeatures(channels=None).to_channel_selection_models() is None
    assert IntensityFeatures(channels=[]).to_channel_selection_models() is None

    ch = InputChannel(mode="index", identifier="0")
    result = IntensityFeatures(channels=[ch]).to_channel_selection_models()
    assert result is not None
    assert len(result) == 1
    assert result[0].identifier == "0"


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
        properties=["label", *ShapeFeatures().property_names(is_2d=True)],
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
        properties=["label", *IntensityFeatures().property_names(is_2d=True)],
    )
    # skimage appends a channel suffix ("-0") when intensity_image has a channel dim
    assert any(k.startswith("intensity_mean") for k in result)


def test_region_props_features_func_wrong_ndim_raises() -> None:
    """Image without channel dim (ndim=2) raises ValueError."""
    image = np.zeros((32, 32), dtype=np.float32)
    label_arr = np.zeros((32, 32), dtype=np.int32)
    with pytest.raises(ValueError):
        region_props_features_func(
            image=image,
            label=label_arr,
            roi=_mock_roi(),
            properties=ShapeFeatures().property_names(is_2d=True),
        )


def test_region_props_features_func_multichannel_label_raises() -> None:
    """Label with C > 1 raises ValueError."""
    image = np.zeros((32, 32, 2), dtype=np.float32)
    label_arr = np.zeros((32, 32, 2), dtype=np.int32)
    with pytest.raises(ValueError):
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
        input_label_name="nuclei",
        features=[ShapeFeatures()],
    )

    some = open_ome_zarr_container(str(store))
    assert "region_props_features" in some.list_tables()


def test_measure_features_shape_and_intensity(tmp_path: Path) -> None:
    """Combining ShapeFeatures and IntensityFeatures succeeds."""
    store = _make_zarr_with_label(tmp_path)

    measure_features(
        zarr_url=str(store),
        input_label_name="nuclei",
        features=[ShapeFeatures(), IntensityFeatures()],
    )

    some = open_ome_zarr_container(str(store))
    assert "region_props_features" in some.list_tables()


def test_measure_features_2d(tmp_path: Path) -> None:
    """Feature extraction works on a 2D (CYX) image."""
    store = _make_zarr_with_label(tmp_path, shape=(1, 32, 32), axes="cyx")

    measure_features(
        zarr_url=str(store),
        input_label_name="nuclei",
        features=[ShapeFeatures()],
    )

    some = open_ome_zarr_container(str(store))
    assert "region_props_features" in some.list_tables()


def test_measure_features_overwrite_true(tmp_path: Path) -> None:
    """Running twice with overwrite=True succeeds."""
    store = _make_zarr_with_label(tmp_path)
    measure_features(
        zarr_url=str(store),
        input_label_name="nuclei",
        features=[ShapeFeatures()],
        overwrite=True,
    )
    measure_features(
        zarr_url=str(store),
        input_label_name="nuclei",
        features=[ShapeFeatures()],
        overwrite=True,
    )


def test_measure_features_overwrite_false_raises(tmp_path: Path) -> None:
    """Running a second time with overwrite=False raises FileExistsError."""
    store = _make_zarr_with_label(tmp_path)

    measure_features(
        zarr_url=str(store),
        input_label_name="nuclei",
        features=[ShapeFeatures()],
        overwrite=True,
    )

    with pytest.raises(FileExistsError):
        measure_features(
            zarr_url=str(store),
            input_label_name="nuclei",
            features=[ShapeFeatures()],
            overwrite=False,
        )


def test_measure_features_custom_table_name(tmp_path: Path) -> None:
    """output_table_name is stored under the given name."""
    store = _make_zarr_with_label(tmp_path)

    measure_features(
        zarr_url=str(store),
        input_label_name="nuclei",
        output_table_name="my_features",
        features=[ShapeFeatures()],
    )

    some = open_ome_zarr_container(str(store))
    assert "my_features" in some.list_tables()


def test_measure_features_advanced_options_no_scaling(tmp_path: Path) -> None:
    """use_scaling=False runs without errors and produces a table."""
    store = _make_zarr_with_label(tmp_path)

    measure_features(
        zarr_url=str(store),
        input_label_name="nuclei",
        features=[ShapeFeatures()],
        advanced_options=AdvancedOptions(use_scaling=False),
    )

    some = open_ome_zarr_container(str(store))
    assert "region_props_features" in some.list_tables()


def test_measure_features_advanced_options_table_backend(tmp_path: Path) -> None:
    """table_backend='parquet' stores the table correctly."""
    store = _make_zarr_with_label(tmp_path)

    measure_features(
        zarr_url=str(store),
        input_label_name="nuclei",
        features=[ShapeFeatures()],
        advanced_options=AdvancedOptions(table_backend="parquet"),
    )

    some = open_ome_zarr_container(str(store))
    assert "region_props_features" in some.list_tables()


def test_intensity_features_with_channels(tmp_path: Path) -> None:
    """IntensityFeatures with explicit channel selection runs without errors."""
    from fractal_tasks_core._measure_features_utils import InputChannel

    store = _make_zarr_with_label(tmp_path)

    measure_features(
        zarr_url=str(store),
        input_label_name="nuclei",
        features=[
            IntensityFeatures(channels=[InputChannel(mode="index", identifier="0")])
        ],
    )

    some = open_ome_zarr_container(str(store))
    assert "region_props_features" in some.list_tables()


# ---------------------------------------------------------------------------
# Correctness tests — measurement table contents
# ---------------------------------------------------------------------------


def test_measure_features_table_structure(tmp_path: Path) -> None:
    """Table has exactly one row, expected columns, and label=1 as index."""
    store = _make_zarr_with_label(tmp_path, shape=(1, 32, 32), axes="cyx")

    measure_features(
        zarr_url=str(store),
        input_label_name="nuclei",
        features=[ShapeFeatures()],
    )

    some = open_ome_zarr_container(str(store))
    df = some.get_table("region_props_features").dataframe
    assert len(df) == 1
    assert "area" in df.columns
    assert "region" in df.columns
    assert df.index.name == "label"
    assert df.index[0] == 1


def test_measure_features_shape_correctness_no_scaling(tmp_path: Path) -> None:
    """num_pixels and area equal the raw pixel count when scaling is disabled."""
    store = _make_zarr_with_label(tmp_path, shape=(1, 32, 32), axes="cyx")

    measure_features(
        zarr_url=str(store),
        input_label_name="nuclei",
        features=[ShapeFeatures()],
        advanced_options=AdvancedOptions(use_scaling=False),
    )

    some = open_ome_zarr_container(str(store))
    df = some.get_table("region_props_features").dataframe
    # Bright region is rows 8:20, cols 8:20 -> 12x12 = 144 pixels
    assert df["num_pixels"].iloc[0] == pytest.approx(144.0)
    assert df["area"].iloc[0] == pytest.approx(144.0)


def test_measure_features_shape_correctness_with_scaling(tmp_path: Path) -> None:
    """area is in physical units (μm²) when scaling is enabled."""
    store = _make_zarr_with_label(tmp_path, shape=(1, 32, 32), axes="cyx")

    measure_features(
        zarr_url=str(store),
        input_label_name="nuclei",
        features=[ShapeFeatures()],
        advanced_options=AdvancedOptions(use_scaling=True),
    )

    some = open_ome_zarr_container(str(store))
    df = some.get_table("region_props_features").dataframe
    # 144 pixels x (0.1 um)**2 = 1.44 um**2
    assert df["area"].iloc[0] == pytest.approx(1.44, rel=1e-3)


def test_measure_features_intensity_correctness(tmp_path: Path) -> None:
    """Intensity statistics are correct for a uniform bright region."""
    store = _make_zarr_with_label(tmp_path, shape=(1, 32, 32), axes="cyx")

    measure_features(
        zarr_url=str(store),
        input_label_name="nuclei",
        features=[IntensityFeatures()],
        advanced_options=AdvancedOptions(use_scaling=False),
    )

    some = open_ome_zarr_container(str(store))
    df = some.get_table("region_props_features").dataframe
    # The bright region is uniformly 500; background (0) is outside the label.
    # Default channel label is "channel_0" (ngio convention).
    assert df["intensity_mean-channel_0"].iloc[0] == pytest.approx(500.0)
    assert df["intensity_max-channel_0"].iloc[0] == pytest.approx(500.0)
    assert df["intensity_min-channel_0"].iloc[0] == pytest.approx(500.0)
    assert df["intensity_std-channel_0"].iloc[0] == pytest.approx(0.0, abs=1e-3)


def test_measure_features_duplicate_feature_type_raises(tmp_path: Path) -> None:
    """Passing the same feature type twice raises ValueError."""
    store = _make_zarr_with_label(tmp_path, shape=(1, 32, 32), axes="cyx")

    with pytest.raises(ValueError, match="Duplicate"):
        measure_features(
            zarr_url=str(store),
            input_label_name="nuclei",
            features=[ShapeFeatures(), ShapeFeatures()],
        )


def test_measure_features_channel_identifier_in_columns(tmp_path: Path) -> None:
    """Explicit channel identifier (index mode) appears in intensity column names."""
    from fractal_tasks_core._measure_features_utils import InputChannel

    store = _make_zarr_with_label(tmp_path, shape=(1, 32, 32), axes="cyx")

    measure_features(
        zarr_url=str(store),
        input_label_name="nuclei",
        features=[
            IntensityFeatures(channels=[InputChannel(mode="index", identifier="0")])
        ],
    )

    some = open_ome_zarr_container(str(store))
    df = some.get_table("region_props_features").dataframe
    assert "intensity_mean-0" in df.columns
    assert "intensity_max-0" in df.columns

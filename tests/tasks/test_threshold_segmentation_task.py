from pathlib import Path

import numpy as np
import pytest
from fractal_tasks_utils.segmentation._transforms import SegmentationTransformConfig
from fractal_tasks_utils.transforms import GaussianBlurConfig, SizeFilterConfig
from ngio import create_empty_ome_zarr, open_ome_zarr_container

from fractal_tasks_core._threshold_segmentation_utils import (
    CreateMaskingRoiTable,
    InputChannel,
    OtsuConfiguration,
    ThresholdConfiguration,
)
from fractal_tasks_core.threshold_segmentation import threshold_segmentation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_czyx_zarr(
    tmp_path: Path,
    shape: tuple = (1, 4, 32, 32),
    name: str = "image",
    bright_value: int = 1000,
) -> Path:
    """Create a CZYX OME-Zarr with a bright rectangular object."""
    store = tmp_path / f"{name}.zarr"
    chunks = tuple(max(s // 2, 1) for s in shape)
    ome_zarr = create_empty_ome_zarr(
        store=store,
        shape=shape,
        chunks=chunks,
        pixelsize=0.1,
        z_spacing=0.5,
        overwrite=False,
        axes_names="czyx",
    )
    image = ome_zarr.get_image()

    data = np.zeros(shape, dtype=np.uint16)
    data[:, :, 10:20, 10:20] = bright_value
    image.set_array(data)
    image.consolidate()
    return store


def _make_cyx_zarr(
    tmp_path: Path,
    shape: tuple = (1, 32, 32),
    name: str = "image_2d",
    bright_value: int = 1000,
) -> Path:
    """Create a CYX OME-Zarr (2D) with a bright rectangular object."""
    store = tmp_path / f"{name}.zarr"
    chunks = tuple(max(s // 2, 1) for s in shape)
    ome_zarr = create_empty_ome_zarr(
        store=store,
        shape=shape,
        chunks=chunks,
        pixelsize=0.1,
        overwrite=False,
        axes_names="cyx",
    )
    image = ome_zarr.get_image()
    data = np.zeros(shape, dtype=np.uint16)
    data[:, 10:20, 10:20] = bright_value
    image.set_array(data)
    image.consolidate()
    return store


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_segmentation_manual_threshold(tmp_path: Path) -> None:
    """Manual threshold creates a label image in the zarr."""
    store = _make_czyx_zarr(tmp_path)

    threshold_segmentation(
        zarr_url=str(store),
        channels=InputChannel(mode="index", identifier="0"),
        output_label_name="nuclei",
        method=ThresholdConfiguration(threshold=500),
        overwrite=True,
    )

    ome = open_ome_zarr_container(str(store))
    assert "nuclei" in ome.list_labels()
    # Check that the label image is correctly thresholded
    nuclei = ome.get_label("nuclei").get_as_numpy()
    labels, counts = np.unique(nuclei, return_counts=True)
    assert set(labels) == {0, 1}  # background and one object
    assert counts[labels == 1][0] == 4 * 10 * 10  # 4*10x10 square of bright pixels


def test_segmentation_otsu(tmp_path: Path) -> None:
    """Otsu threshold on data with a bright object creates a label."""
    store = _make_czyx_zarr(tmp_path)

    threshold_segmentation(
        zarr_url=str(store),
        channels=InputChannel(mode="index", identifier="0"),
        output_label_name="nuclei",
        method=OtsuConfiguration(),
        overwrite=True,
    )

    ome = open_ome_zarr_container(str(store))
    assert "nuclei" in ome.list_labels()
    nuclei = ome.get_label("nuclei").get_as_numpy()
    labels, counts = np.unique(nuclei, return_counts=True)
    assert set(labels) == {0, 1}  # background and one object
    assert counts[labels == 1][0] == 4 * 10 * 10  # 4*10x10 square of bright pixels


def test_segmentation_2d(tmp_path: Path) -> None:
    """Segmentation works on a 2D (CYX) image."""
    store = _make_cyx_zarr(tmp_path)

    threshold_segmentation(
        zarr_url=str(store),
        channels=InputChannel(mode="index", identifier="0"),
        output_label_name="nuclei",
        method=ThresholdConfiguration(threshold=500),
        overwrite=True,
    )

    ome = open_ome_zarr_container(str(store))
    assert "nuclei" in ome.list_labels()
    nuclei = ome.get_label("nuclei").get_as_numpy()
    labels, counts = np.unique(nuclei, return_counts=True)
    assert set(labels) == {0, 1}  # background and one object
    assert counts[labels == 1][0] == 10 * 10  # 4*10x10 square of bright pixels


def test_label_name_template(tmp_path: Path) -> None:
    """{channel_identifier} in label_name is replaced with the channel identifier."""
    store = _make_czyx_zarr(tmp_path)

    threshold_segmentation(
        zarr_url=str(store),
        channels=InputChannel(mode="index", identifier="0"),
        output_label_name="{channel_identifier}_nuclei",
        method=ThresholdConfiguration(threshold=500),
        overwrite=True,
    )

    ome = open_ome_zarr_container(str(store))
    assert "0_nuclei" in ome.list_labels()


def test_invalid_label_name_raises(tmp_path: Path) -> None:
    """An unsupported placeholder in the label name raises ValueError."""
    store = _make_czyx_zarr(tmp_path)

    with pytest.raises(ValueError, match="channel_identifier"):
        threshold_segmentation(
            zarr_url=str(store),
            channels=InputChannel(mode="index", identifier="0"),
            output_label_name="{bad_placeholder}_nuclei",
            method=ThresholdConfiguration(threshold=500),
            overwrite=True,
        )


def test_skip_missing_channel_true(tmp_path: Path) -> None:
    """skip_if_missing=True returns early without creating a label."""
    store = _make_czyx_zarr(tmp_path)

    threshold_segmentation(
        zarr_url=str(store),
        channels=InputChannel(
            mode="label", identifier="does_not_exist", skip_if_missing=True
        ),
        output_label_name="nuclei",
        method=ThresholdConfiguration(threshold=500),
        overwrite=True,
    )

    ome = open_ome_zarr_container(str(store))
    assert "nuclei" not in ome.list_labels()


def test_missing_channel_raises(tmp_path: Path) -> None:
    """skip_if_missing=False raises ValueError when the channel is absent."""
    store = _make_czyx_zarr(tmp_path)

    with pytest.raises(ValueError):
        threshold_segmentation(
            zarr_url=str(store),
            channels=InputChannel(
                mode="label", identifier="does_not_exist", skip_if_missing=False
            ),
            output_label_name="nuclei",
            method=ThresholdConfiguration(threshold=500),
            overwrite=True,
        )


def test_overwrite_true(tmp_path: Path) -> None:
    """Running twice with overwrite=True both succeed."""
    store = _make_czyx_zarr(tmp_path)
    threshold_segmentation(
        zarr_url=str(store),
        channels=InputChannel(mode="index", identifier="0"),
        output_label_name="nuclei",
        method=ThresholdConfiguration(threshold=500),
        overwrite=True,
    )
    threshold_segmentation(
        zarr_url=str(store),
        channels=InputChannel(mode="index", identifier="0"),
        output_label_name="nuclei",
        method=ThresholdConfiguration(threshold=500),
        overwrite=True,
    )


def test_overwrite_false_raises(tmp_path: Path) -> None:
    """Running a second time with overwrite=False raises."""
    store = _make_czyx_zarr(tmp_path)
    threshold_segmentation(
        zarr_url=str(store),
        channels=InputChannel(mode="index", identifier="0"),
        output_label_name="nuclei",
        method=ThresholdConfiguration(threshold=500),
        overwrite=True,
    )

    with pytest.raises(Exception):
        threshold_segmentation(
            zarr_url=str(store),
            channels=InputChannel(mode="index", identifier="0"),
            output_label_name="nuclei",
            method=ThresholdConfiguration(threshold=500),
            overwrite=False,
        )


def test_create_masking_roi_table(tmp_path: Path) -> None:
    """CreateMaskingRoiTable creates a table with the derived name."""
    store = _make_czyx_zarr(tmp_path)

    threshold_segmentation(
        zarr_url=str(store),
        channels=InputChannel(mode="index", identifier="0"),
        output_label_name="nuclei",
        method=ThresholdConfiguration(threshold=500),
        create_masking_roi_table=CreateMaskingRoiTable(),
        overwrite=True,
    )

    ome = open_ome_zarr_container(str(store))
    assert "nuclei_masking_ROI_table" in ome.list_tables()


def test_with_gaussian_preprocess(tmp_path: Path) -> None:
    """Gaussian pre-processing does not prevent label creation."""
    store = _make_czyx_zarr(tmp_path)
    config = SegmentationTransformConfig(pre_process=[GaussianBlurConfig(sigma_xy=1.0)])

    threshold_segmentation(
        zarr_url=str(store),
        channels=InputChannel(mode="index", identifier="0"),
        output_label_name="nuclei",
        method=ThresholdConfiguration(threshold=500),
        pre_post_process=config,
        overwrite=True,
    )

    ome = open_ome_zarr_container(str(store))
    assert "nuclei" in ome.list_labels()


def test_with_size_filter_postprocess(tmp_path: Path) -> None:
    """SizeFilter post-processing runs end-to-end."""
    store = _make_czyx_zarr(tmp_path)
    config = SegmentationTransformConfig(post_process=[SizeFilterConfig(min_size=5)])

    threshold_segmentation(
        zarr_url=str(store),
        channels=InputChannel(mode="index", identifier="0"),
        output_label_name="nuclei",
        method=ThresholdConfiguration(threshold=500),
        pre_post_process=config,
        overwrite=True,
    )

    ome = open_ome_zarr_container(str(store))
    assert "nuclei" in ome.list_labels()

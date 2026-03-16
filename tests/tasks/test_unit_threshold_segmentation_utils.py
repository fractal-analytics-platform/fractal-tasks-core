import numpy as np
import pytest

from fractal_tasks_core._threshold_segmentation_utils import (
    CreateMaskingRoiTable,
    OtsuConfiguration,
    SimpleThresholdConfiguration,
    segmentation_function,
)
from fractal_tasks_core._utils import format_template_name

# ---------------------------------------------------------------------------
# ThresholdConfiguration / OtsuConfiguration
# ---------------------------------------------------------------------------


def test_threshold_configuration_value() -> None:
    config = SimpleThresholdConfiguration(threshold=42.0)
    img = np.zeros((10, 10), dtype=np.float32)
    assert config.threshold_value(img) == 42.0


def test_otsu_configuration_value() -> None:
    config = OtsuConfiguration()
    img = np.zeros((32, 32), dtype=np.float32)
    img[10:20, 10:20] = 100.0
    thresh = config.threshold_value(img)
    assert 0.0 < thresh < 100.0


# ---------------------------------------------------------------------------
# segmentation_function
# ---------------------------------------------------------------------------


def test_segmentation_function_manual_threshold() -> None:
    """Manual threshold on a 3D image produces a labeled output with correct shape."""
    img = np.zeros((4, 32, 32), dtype=np.float32)
    img[1:3, 10:20, 10:20] = 100.0
    result = segmentation_function(
        input_image=img,
        method=SimpleThresholdConfiguration(threshold=50.0),
    )
    assert result.shape == (4, 32, 32)
    assert result.dtype == np.uint32
    assert result.max() > 0


def test_segmentation_function_otsu() -> None:
    """Otsu threshold on a 2D image with bright object produces a label."""
    img = np.zeros((32, 32), dtype=np.float32)
    img[10:20, 10:20] = 100.0
    result = segmentation_function(
        input_image=img,
        method=OtsuConfiguration(),
    )
    assert result.shape == (32, 32)
    assert result.max() > 0


# ---------------------------------------------------------------------------
# CreateMaskingRoiTable.get_table_name
# ---------------------------------------------------------------------------


def test_create_masking_roi_table_get_name_default() -> None:
    conf = CreateMaskingRoiTable()
    assert conf.get_table_name("nuclei") == "nuclei_masking_ROI_table"


def test_create_masking_roi_table_get_name_custom() -> None:
    conf = CreateMaskingRoiTable(table_name="{label_name}_roi")
    assert conf.get_table_name("cells") == "cells_roi"


# ---------------------------------------------------------------------------
# format_template_name
# ---------------------------------------------------------------------------


def test_format_template_name_with_placeholder() -> None:
    result = format_template_name("{channel_identifier}_seg", channel_identifier="DAPI")
    assert result == "DAPI_seg"


def test_format_template_name_without_placeholder() -> None:
    result = format_template_name("nuclei", channel_identifier="DAPI")
    assert result == "nuclei"


def test_format_template_name_invalid_placeholder() -> None:
    with pytest.raises(ValueError, match="channel_identifier"):
        format_template_name("{bad_key}_seg", channel_identifier="DAPI")

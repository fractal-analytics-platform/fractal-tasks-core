import json
from pathlib import Path

import numpy as np
import pytest
from devtools import debug

"""
from fractal_tasks_core.lib_zattrs_utils import extract_zyx_pixel_sizes
from fractal_tasks_core.lib_zattrs_utils import (
    get_acquisition_paths,
)
"""
from fractal_tasks_core.lib_ngff import Dataset
from fractal_tasks_core.lib_ngff import Multiscale
from fractal_tasks_core.lib_ngff import NgffImageMeta
from fractal_tasks_core.lib_ngff import NgffWellMeta


def _load_and_validate(path, Model):
    with open(path, "r") as f:
        data = json.load(f)
    return Model(**data)


@pytest.fixture
def ngffdata_path(testdata_path: Path) -> Path:
    return testdata_path / "ngff_examples"


def test_Dataset(ngffdata_path):
    _load_and_validate(ngffdata_path / "dataset.json", Dataset)

    # Fail due to missing scale transformation
    dataset = _load_and_validate(
        ngffdata_path / "dataset_error_1.json", Dataset
    )
    with pytest.raises(ValueError) as e:
        dataset.scale_transformation
    assert "Missing scale transformation" in str(e.value)

    # Fail due to multiple scale transformations
    dataset = _load_and_validate(
        ngffdata_path / "dataset_error_2.json", Dataset
    )
    with pytest.raises(ValueError) as e:
        dataset.scale_transformation
    assert "More than one scale transformation" in str(e.value)


def test_Multiscale(ngffdata_path):
    # Fail due to global coordinateTransformation
    with pytest.raises(NotImplementedError):
        _load_and_validate(ngffdata_path / "multiscale_error.json", Multiscale)

    # Success
    _load_and_validate(ngffdata_path / "multiscale.json", Multiscale)


def test_NgffImageMeta(ngffdata_path):

    # Fail when accessing multiscale, if there are more than one
    ngff_image_meta = _load_and_validate(
        ngffdata_path / "image_error.json", NgffImageMeta
    )
    with pytest.raises(NotImplementedError):
        ngff_image_meta.multiscale

    # Success CZYX
    ngff_image_meta = _load_and_validate(
        ngffdata_path / "image.json", NgffImageMeta
    )
    assert ngff_image_meta.multiscale
    assert len(ngff_image_meta.datasets) == 5
    assert len(ngff_image_meta.datasets) == ngff_image_meta.num_levels
    assert ngff_image_meta.axes == ["c", "z", "y", "x"]
    assert np.allclose(
        ngff_image_meta.get_pixel_sizes_zyx(), [1.0, 0.1625, 0.1625]
    )
    assert np.allclose(
        ngff_image_meta.get_pixel_sizes_zyx(level=0), [1.0, 0.1625, 0.1625]
    )
    assert np.allclose(
        ngff_image_meta.get_pixel_sizes_zyx(level=1), [1.0, 0.325, 0.325]
    )
    assert ngff_image_meta.coarsening_xy == 2

    # Success ZYX
    ngff_image_meta = _load_and_validate(
        ngffdata_path / "image_ZYX.json", NgffImageMeta
    )
    assert ngff_image_meta.multiscale
    assert len(ngff_image_meta.datasets) == 5
    assert len(ngff_image_meta.datasets) == ngff_image_meta.num_levels
    assert ngff_image_meta.axes == ["z", "y", "x"]
    assert np.allclose(
        ngff_image_meta.get_pixel_sizes_zyx(), [1.0, 0.1625, 0.1625]
    )
    assert np.allclose(
        ngff_image_meta.get_pixel_sizes_zyx(level=0), [1.0, 0.1625, 0.1625]
    )
    assert np.allclose(
        ngff_image_meta.get_pixel_sizes_zyx(level=1), [1.0, 0.325, 0.325]
    )
    assert ngff_image_meta.coarsening_xy == 2

    # Pixel sizes are too small
    ngff_image_meta = _load_and_validate(
        ngffdata_path / "image_error_pixels.json", NgffImageMeta
    )
    with pytest.raises(ValueError) as e:
        ngff_image_meta.pixel_sizes_zyx
    debug(e.value)
    assert "are too small" in str(e.value)


def test_ImageNgffMeta_missing_Z(ngffdata_path, caplog):
    ngff_image_meta = _load_and_validate(
        ngffdata_path / "image_CYX.json", NgffImageMeta
    )
    caplog.clear()
    ngff_image_meta.pixel_sizes_zyx
    debug(caplog.text)
    assert "Z axis is not present" in caplog.text


def test_ImageNgffMeta_inhomogeneous_coarsening(ngffdata_path):
    # Case 1
    ngff_image_meta = _load_and_validate(
        ngffdata_path / "image_error_coarsening_1.json", NgffImageMeta
    )
    with pytest.raises(NotImplementedError) as e:
        ngff_image_meta.coarsening_xy
    assert "Inhomogeneous coarsening in X/Y directions" in str(e.value)
    # Case 2
    ngff_image_meta = _load_and_validate(
        ngffdata_path / "image_error_coarsening_2.json", NgffImageMeta
    )
    with pytest.raises(NotImplementedError) as e:
        ngff_image_meta.coarsening_xy
    assert "Inhomogeneous coarsening across levels" in str(e.value)


def test_NgffWellMeta_get_acquisition_paths(ngffdata_path):
    # Fail for no acquisition keys
    ngff_well_meta = _load_and_validate(
        ngffdata_path / "well.json", NgffWellMeta
    )
    with pytest.raises(ValueError) as e:
        ngff_well_meta.get_acquisition_paths()
    assert "Cannot get acquisition paths" in str(e.value)

    # Fail for repeated acquisitions
    ngff_well_meta = _load_and_validate(
        ngffdata_path / "well_acquisitions_error.json", NgffWellMeta
    )
    with pytest.raises(NotImplementedError) as e:
        ngff_well_meta.get_acquisition_paths()
    assert "multiple images of the same acquisition" in str(e.value)

    # Success
    ngff_well_meta = _load_and_validate(
        ngffdata_path / "well_acquisitions.json", NgffWellMeta
    )
    debug(ngff_well_meta.get_acquisition_paths())
    assert ngff_well_meta.get_acquisition_paths() == {9: "nine", 7: "seven"}

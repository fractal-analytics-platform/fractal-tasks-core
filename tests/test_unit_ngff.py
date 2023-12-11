import json
from pathlib import Path

import numpy as np
import pytest
import zarr
from devtools import debug

from fractal_tasks_core.ome_zarr.ngff import Dataset
from fractal_tasks_core.ome_zarr.ngff import detect_ome_ngff_type
from fractal_tasks_core.ome_zarr.ngff import load_NgffImageMeta
from fractal_tasks_core.ome_zarr.ngff import load_NgffWellMeta
from fractal_tasks_core.ome_zarr.ngff import Multiscale
from fractal_tasks_core.ome_zarr.ngff import NgffImageMeta
from fractal_tasks_core.ome_zarr.ngff import NgffWellMeta
from fractal_tasks_core.ome_zarr.ngff import ZarrGroupNotFoundError


def test_load_NgffWellMeta(tmp_path):
    path = str(tmp_path / "error.zarr")
    group = zarr.open_group(path)
    group.attrs.put({"something": "else"})
    with pytest.raises(ValueError) as e:
        load_NgffImageMeta(path)
    debug(e.value)


def test_load_NgffImageMeta_missing_group(tmp_path):
    path = str(tmp_path / "missing.zarr")
    with pytest.raises(ZarrGroupNotFoundError) as e:
        load_NgffImageMeta(path)
    assert "Could not load attributes for the requested image" in str(e.value)
    assert path in str(e.value)


def test_load_NgffWellMeta_missing_group(tmp_path):
    path = str(tmp_path / "missing.zarr")
    with pytest.raises(ZarrGroupNotFoundError) as e:
        load_NgffWellMeta(path)
    assert "Could not load attributes for the requested well" in str(e.value)
    assert path in str(e.value)


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
    assert ngff_image_meta.axes_names == ["c", "z", "y", "x"]
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
    assert ngff_image_meta.axes_names == ["z", "y", "x"]
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

    # Load an image where not all omero channels have a window attribute
    ngff_image_meta = _load_and_validate(
        ngffdata_path / "image_no_omero_channel_window.json", NgffImageMeta
    )
    debug(ngff_image_meta.omero)
    assert ngff_image_meta.omero.channels[0].window is None
    assert ngff_image_meta.omero.channels[1].window is not None


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


def test_detect_ome_ngff_type(tmp_path):

    g_plate = zarr.open(tmp_path / "plate.zarr")
    g_plate.attrs.update(plate={})
    g_well = zarr.open(tmp_path / "well.zarr")
    g_well.attrs.update(well={})
    g_image = zarr.open(tmp_path / "image.zarr")
    g_image.attrs.update(multiscales=[])
    g_wrong = zarr.open(tmp_path / "wrong.zarr")
    g_wrong.attrs.update(something="else")

    assert detect_ome_ngff_type(g_plate) == "plate"
    assert detect_ome_ngff_type(g_well) == "well"
    assert detect_ome_ngff_type(g_image) == "image"
    with pytest.raises(ValueError):
        detect_ome_ngff_type(g_wrong)

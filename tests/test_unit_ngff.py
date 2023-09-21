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

# from fractal_tasks_core.lib_ngff import NgffWellMeta


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

    # Success
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


"""
    # Case 2: fail for global coordinateTransformations
    metadata = dict(multiscales=[dict(axes=[], coordinateTransformations=[])])
    with pytest.raises(NotImplementedError) as e:
        _call_extract_zyx_pixel_sizes(metadata)
    debug(e.value)
    assert "Global coordinateTransformations" in str(e.value)

    # Case 3: fail for missing scale transformation
    metadata = dict(
        multiscales=[
            dict(axes=[], datasets=[dict(coordinateTransformations=[])])
        ]
    )
    with pytest.raises(ValueError) as e:
        _call_extract_zyx_pixel_sizes(metadata)
    debug(e.value)
    assert "No scale transformation found" in str(e.value)

    # Case 4: success, with 4D (CZXY) scale transformation
    metadata = dict(
        multiscales=[
            dict(
                axes=[],
                datasets=[
                    dict(
                        coordinateTransformations=[
                            dict(type="scale", scale=[1, 2, 2, 2])
                        ]
                    )
                ],
            )
        ]
    )
    out = _call_extract_zyx_pixel_sizes(metadata)
    debug(out)
    assert out == [2, 2, 2]

    # Case 5: success, with 3D (ZYX) scale transformation
    metadata = dict(
        multiscales=[
            dict(
                axes=[],
                datasets=[
                    dict(
                        coordinateTransformations=[
                            dict(type="scale", scale=[2, 2, 2])
                        ]
                    )
                ],
            )
        ]
    )
    out = _call_extract_zyx_pixel_sizes(metadata)
    debug(out)
    assert out == [2, 2, 2]

    # Case 6: fail because pixel sizes are too small
    metadata = dict(
        multiscales=[
            dict(
                axes=[],
                datasets=[
                    dict(
                        coordinateTransformations=[
                            dict(type="scale", scale=[2, 2, 1e-20])
                        ]
                    )
                ],
            )
        ]
    )
    with pytest.raises(ValueError) as e:
        _call_extract_zyx_pixel_sizes(metadata)
    debug(e.value)


def test_get_acquisition_paths():

    # FIXME: transform into unit testing of ngff classes/methods

    # Successful call
    image_1 = dict(path="path1", acquisition=1)
    image_2 = dict(path="path2", acquisition=2)
    zattrs = dict(well=dict(images=[image_1, image_2]))
    res = get_acquisition_paths(zattrs)
    debug(res)
    assert res == {1: "path1", 2: "path2"}

    # Fail (missing acquisition key)
    image_1 = dict(path="path1", acquisition=1)
    image_2 = dict(path="path2")
    zattrs = dict(well=dict(images=[image_1, image_2]))
    with pytest.raises(ValueError):
        get_acquisition_paths(zattrs)

    # Fail (non-unique acquisition value)
    image_1 = dict(path="path1", acquisition=1)
    image_2 = dict(path="path2", acquisition=1)
    zattrs = dict(well=dict(images=[image_1, image_2]))
    with pytest.raises(NotImplementedError):
        get_acquisition_paths(zattrs)
"""

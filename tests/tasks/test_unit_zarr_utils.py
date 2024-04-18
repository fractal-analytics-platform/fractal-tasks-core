import logging
import shutil
from pathlib import Path

import pytest
import zarr
from devtools import debug
from pytest import LogCaptureFixture

from fractal_tasks_core.ngff.zarr_utils import load_NgffWellMeta
from fractal_tasks_core.tasks._registration_utils import (
    _split_well_path_image_path,
)
from fractal_tasks_core.tasks._zarr_utils import _copy_hcs_ome_zarr_metadata


@pytest.mark.parametrize("trailing_slash", [True, False])
def test_copy_hcs_ome_zarr_metadata(
    tmp_path: Path,
    testdata_path: Path,
    caplog: LogCaptureFixture,
    trailing_slash: bool,
):
    caplog.set_level(logging.INFO)

    # Copy a reference zarr into a temporary folder
    raw_zarrurl = (testdata_path / "plate_ones.zarr").as_posix()
    zarr_url = (tmp_path / "plate.zarr").resolve().as_posix()
    shutil.copytree(raw_zarrurl, zarr_url)
    zarr_url += "/B/03/0"
    suffix = "_illum_corr"
    new_zarr_url = zarr_url + suffix
    if trailing_slash:
        zarr_url += "/"
        new_zarr_url += "/"

    _copy_hcs_ome_zarr_metadata(
        zarr_url_origin=zarr_url, zarr_url_new=new_zarr_url
    )

    group = zarr.open_group(zarr_url, mode="r")
    old_attrs = group.attrs.asdict()
    group_new = zarr.open_group(new_zarr_url, mode="r")
    new_attrs = group_new.attrs.asdict()
    debug(old_attrs)
    assert old_attrs == new_attrs

    # Check well metadata:
    well_url, _ = _split_well_path_image_path(zarr_url=zarr_url)
    well_meta = load_NgffWellMeta(well_url)
    debug(well_meta)
    well_paths = [image.path for image in well_meta.well.images]
    assert well_paths == ["0", "0" + suffix]

import logging
import multiprocessing as mp
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest
import zarr
from devtools import debug
from filelock._error import Timeout
from pytest import LogCaptureFixture

from fractal_tasks_core.ngff.zarr_utils import load_NgffWellMeta
from fractal_tasks_core.tasks._registration_utils import (
    _split_well_path_image_path,
)
from fractal_tasks_core.tasks._zarr_utils import _copy_hcs_ome_zarr_metadata
from fractal_tasks_core.tasks._zarr_utils import (
    _get_matching_ref_acquisition_path_heuristic,
)
from fractal_tasks_core.tasks._zarr_utils import _update_well_metadata


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


def _star_update_well_metadata(args):
    """
    This is only needed because concurrent.futures executors have a `map`
    method but not a `starmap` one.
    """
    return _update_well_metadata(*args)


INTERVAL = 0.5


def _slow_load_NgffWellMeta(*args, **kwargs):
    logging.warning(
        f"START _slow wrapper for {args}, {time.perf_counter():.3f}"
    )
    time.sleep(INTERVAL)
    output = load_NgffWellMeta(*args, **kwargs)
    logging.warning(f"END _slow wrapper for {args}, {time.perf_counter():.3f}")
    return output


def test_update_well_metadata_concurrency(
    tmp_path: Path,
    testdata_path: Path,
    monkeypatch,
):
    """
    Run _update_well_metadata in parallel for adding N>1 new images to a given
    well. We artificially slow down each call by INTERVAL seconds, and verify
    that the test takes at least N x INTERVAL seconds (since each call to
    `_update_well_metadata` is blocking).

    In the last section of the test, we verify that a timeout error is raised
    when the timeout is too short.
    """

    N = 4

    # Copy a reference zarr into a temporary folder
    raw_zarrurl = (testdata_path / "plate_ones.zarr").as_posix()
    zarr_url = (tmp_path / "plate.zarr").resolve().as_posix()
    shutil.copytree(raw_zarrurl, zarr_url)

    # Artificially slow down `_update_well_metadata`
    import fractal_tasks_core.tasks._zarr_utils

    monkeypatch.setattr(
        fractal_tasks_core.tasks._zarr_utils,
        "load_NgffWellMeta",
        _slow_load_NgffWellMeta,
    )

    # Prepare parallel-execution argument list
    well_url = Path(zarr_url, "B/03").as_posix()
    list_args = [(well_url, "0", f"0_new_{suffix}") for suffix in range(N)]

    # Run `_update_well_metadata` N times
    time_start = time.perf_counter()
    with ProcessPoolExecutor(mp_context=mp.get_context("fork")) as executor:
        res_iter = executor.map(_star_update_well_metadata, list_args)
        list(res_iter)  # This is needed, to wait for all results.
    time_end = time.perf_counter()

    # Check that time was at least N*INTERVAL seconds
    assert (time_end - time_start) > N * INTERVAL

    # Check that all new images were added
    well_meta = load_NgffWellMeta(well_url)
    well_image_paths = [img.path for img in well_meta.well.images]
    debug(well_image_paths)
    assert well_image_paths == [
        "0",
        "0_new_0",
        "0_new_1",
        "0_new_2",
        "0_new_3",
    ]

    # Prepare parallel-execution argument list with short timeout
    well_url = Path(zarr_url, "B/03").as_posix()
    list_args = [
        (well_url, "0", f"0_new_{suffix}", INTERVAL / 100)
        for suffix in range(N, 2 * N)
    ]
    with pytest.raises(Timeout) as e:
        with ProcessPoolExecutor(
            mp_context=mp.get_context("fork")
        ) as executor:
            res_iter = executor.map(_star_update_well_metadata, list_args)
            list(res_iter)  # This is needed, to wait for all results.
    debug(e.value)


def test_update_well_metadata_failures(
    tmp_path: Path,
    testdata_path: Path,
):
    """
    When called with an invalid `old_image_path` or `new_image_path`,
    `_update_well_metadata` fails as expected.
    """

    # Copy a reference zarr into a temporary folder
    raw_zarrurl = (testdata_path / "plate_ones.zarr").as_posix()
    zarr_url = (tmp_path / "plate.zarr").resolve().as_posix()
    shutil.copytree(raw_zarrurl, zarr_url)
    well_url = Path(zarr_url, "B/03").as_posix()

    # Failure case 1
    with pytest.raises(ValueError) as e:
        _update_well_metadata(well_url, "INVALID_OLD_IMAGE_PATH", "0_new")

    assert "Could not find an image with old_image_path" in str(e.value)

    # Failure case 2
    with pytest.raises(ValueError) as e:
        _update_well_metadata(well_url, "0", "0")

    assert "Could not add the new_image_path" in str(e.value)


HEURISTIC_CASES = [
    (["0", "0_illum_corr"], "1_illum_corr", "0_illum_corr"),
    (["0", "0_illum_corr"], "1", "0"),
    (["0", "0_illum_corr", "0_registered"], "1_illum_corr", "0_illum_corr"),
    (["0", "1", "2", "3"], "0_acq2", "0"),
    (["1", "0", "2", "3"], "0_acq2", "0"),
    (["0", "1", "2", "3"], "3", "0"),
]


@pytest.mark.parametrize("path_list, path, expected_match", HEURISTIC_CASES)
def test_get_matching_ref_acquisition_path_heuristic(
    path_list: list[str], path: str, expected_match: str
):
    match = _get_matching_ref_acquisition_path_heuristic(
        path_list=path_list, path=path
    )
    assert match == expected_match

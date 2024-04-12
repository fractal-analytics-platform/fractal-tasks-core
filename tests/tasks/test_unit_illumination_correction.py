import json
import logging
import shutil
from pathlib import Path

import anndata as ad
import dask.array as da
import numpy as np
from pytest import LogCaptureFixture
from pytest import MonkeyPatch

from fractal_tasks_core.ngff.zarr_utils import load_NgffImageMeta
from fractal_tasks_core.roi import (
    convert_ROI_table_to_indices,
)
from fractal_tasks_core.tasks.illumination_correction import correct
from fractal_tasks_core.tasks.illumination_correction import (
    illumination_correction,
)


def test_illumination_correction(
    tmp_path: Path,
    testdata_path: Path,
    monkeypatch: MonkeyPatch,
    caplog: LogCaptureFixture,
):
    # GIVEN a zarr pyramid on disk, made of all ones
    # WHEN I apply illumination_correction
    # THEN correct(..) is executed as many times as
    #      (number of FOVs) x (number of channels)
    # AND the output array has ones at all pyramid levels

    caplog.set_level(logging.INFO)

    # Copy a reference zarr into a temporary folder
    raw_zarrurl = (testdata_path / "plate_ones.zarr").as_posix()
    zarr_url = (tmp_path / "plate.zarr").resolve().as_posix()
    shutil.copytree(raw_zarrurl, zarr_url)
    zarr_url += "/B/03/0/"

    # Prepare arguments for illumination_correction function
    testdata_str = testdata_path.as_posix()
    illum_params = {
        "A01_C01": "illum_corr_matrix.png",
        "A01_C02": "illum_corr_matrix.png",
    }
    illumination_profiles_folder = f"{testdata_str}/illumination_correction/"
    with open(zarr_url + ".zattrs") as fin:
        zattrs = json.load(fin)
        num_levels = len(zattrs["multiscales"][0]["datasets"])
    metadata: dict = {
        "num_levels": num_levels,
        "coarsening_xy": 2,
    }
    num_channels = 2
    num_levels = metadata["num_levels"]

    # Read FOV ROIs and create corresponding indices
    ngff_image_meta = load_NgffImageMeta(zarr_url)
    pixels = ngff_image_meta.get_pixel_sizes_zyx(level=0)
    ROIs = ad.read_zarr(zarr_url + "tables/FOV_ROI_table/")
    list_indices = convert_ROI_table_to_indices(
        ROIs, level=0, full_res_pxl_sizes_zyx=pixels
    )
    num_FOVs = len(list_indices)

    # Prepared expected number of calls
    expected_tot_calls_correct = num_channels * num_FOVs

    # Patch correct() function, to keep track of the number of calls
    logfile = (tmp_path / "log_function_correct.txt").resolve().as_posix()
    with open(logfile, "w") as log:
        log.write("")

    def patched_correct(*args, **kwargs):
        with open(logfile, "a") as log:
            log.write("1\n")
        return correct(*args, **kwargs)

    monkeypatch.setattr(
        "fractal_tasks_core.tasks.illumination_correction.correct",
        patched_correct,
    )

    # Call illumination correction task, with patched correct()
    illumination_correction(
        zarr_url=zarr_url,
        overwrite_input=True,
        illumination_profiles_folder=illumination_profiles_folder,
        dict_corr=illum_params,
        background=0,
    )

    print(caplog.text)
    caplog.clear()

    # Verify the total number of calls
    with open(logfile, "r") as f:
        tot_calls_correct = len(f.read().splitlines())
    assert tot_calls_correct == expected_tot_calls_correct

    # Verify the output
    for ind_level in range(num_levels):
        old = da.from_zarr(
            testdata_path / f"plate_ones.zarr/B/03/0/{ind_level}"
        )
        new = da.from_zarr(f"{zarr_url}{ind_level}")
        assert old.shape == new.shape
        assert old.chunks == new.chunks
        assert new.compute()[0, 0, 0, 0] == 1
        assert np.allclose(old.compute(), new.compute())

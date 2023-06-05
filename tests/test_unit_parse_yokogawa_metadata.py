"""
Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
University of Zurich

Original authors:
Joel Lüthi  <joel.luethi@fmi.ch>

This file is part of Fractal and was originally developed by eXact lab S.r.l.
<exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
Institute for Biomedical Research and Pelkmans Lab from the University of
Zurich.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from devtools import debug
from pandas import Timestamp

from fractal_tasks_core.dev.lib_metadata_checks import run_overlap_check
from fractal_tasks_core.lib_metadata_parsing import parse_yokogawa_metadata
from fractal_tasks_core.lib_ROI_overlaps import remove_FOV_overlaps

# General variables and paths (relative to the test folder)
testdir = os.path.dirname(__file__)

path = f"{testdir}/data/metadata_files/"

# Test set 1
mlf_path_1 = f"{path}MeasurementData_SingleWell2Sites_MultiZ.mlf"
mrf_path_1 = f"{path}MeasurementDetail.mrf"
expected_files_1 = 4
expected_shape_1 = (2, 11)
x_mic_pos_1 = (-1448.3, -1032.3)
y_mic_pos_1 = -1517.7
pixel_size_z_1 = 1.0
z_pixel_1 = 2
pixel_size_x = 0.1625
pixel_size_y = 0.1625
x_pixel = 2560
y_pixel = 2160
bit_depth = 16
time_1 = [
    Timestamp("2020-08-12 15:36:36.234000+0000", tz="UTC"),
    Timestamp("2020-08-12 15:36:46.322000+0000", tz="UTC"),
]

# Test set 2
mlf_path_2 = f"{path}MeasurementData_2x2_well.mlf"
mrf_path_2 = f"{path}MeasurementDetail_2x2_well.mrf"
expected_files_2 = 120
expected_shape_2 = (4, 11)
x_mic_pos_2 = (-1448.3, -1032.3)
y_mic_pos_2 = (-1517.7, -1166.7)
pixel_size_z_2 = 1.0
z_pixel_2 = 10
time_2 = [
    Timestamp("2020-08-12 15:36:36.234000+0000", tz="UTC"),
    Timestamp("2020-08-12 15:36:46.322000+0000", tz="UTC"),
    Timestamp("2020-08-12 15:37:51.787000+0000", tz="UTC"),
    Timestamp("2020-08-12 15:38:01.703000+0000", tz="UTC"),
]

# Test set 3
mlf_path_3 = f"{path}MeasurementData_SingleWell2Sites_SingleZ.mlf"
mrf_path_3 = f"{path}MeasurementDetail.mrf"
expected_files_3 = 2
expected_shape_3 = (2, 11)
x_mic_pos_3 = (-1448.3, -1032.3)
y_mic_pos_3 = -1517.7
pixel_size_z_3 = 1.0
z_pixel_3 = 1
time_3 = [
    Timestamp("2020-08-12 15:36:36.234000+0000", tz="UTC"),
    Timestamp("2020-08-12 15:36:46.322000+0000", tz="UTC"),
]


parameters = [
    (
        mlf_path_1,
        mrf_path_1,
        expected_files_1,
        expected_shape_1,
        x_mic_pos_1,
        y_mic_pos_1,
        pixel_size_z_1,
        z_pixel_1,
        pixel_size_x,
        pixel_size_y,
        x_pixel,
        y_pixel,
        bit_depth,
        time_1,
    ),
    (
        mlf_path_2,
        mrf_path_2,
        expected_files_2,
        expected_shape_2,
        x_mic_pos_2,
        y_mic_pos_2,
        pixel_size_z_2,
        z_pixel_2,
        pixel_size_x,
        pixel_size_y,
        x_pixel,
        y_pixel,
        bit_depth,
        time_2,
    ),
    (
        mlf_path_3,
        mrf_path_3,
        expected_files_3,
        expected_shape_3,
        x_mic_pos_3,
        y_mic_pos_3,
        pixel_size_z_3,
        z_pixel_3,
        pixel_size_x,
        pixel_size_y,
        x_pixel,
        y_pixel,
        bit_depth,
        time_3,
    ),
]


@pytest.mark.parametrize(
    "mlf_path, mrf_path, expected_files, expected_shape, x_mic_pos, "
    "y_mic_pos, pixel_size_z, z_pixel, pixel_size_x, pixel_size_y, "
    "x_pixel, y_pixel, bit_depth, Time",
    parameters,
)
def test_parse_yokogawa_metadata(
    mlf_path,
    mrf_path,
    expected_files,
    expected_shape,
    x_mic_pos,
    y_mic_pos,
    pixel_size_z,
    z_pixel,
    pixel_size_x,
    pixel_size_y,
    x_pixel,
    y_pixel,
    bit_depth,
    Time,
):
    site_metadata, total_files = parse_yokogawa_metadata(mrf_path, mlf_path)
    debug(total_files)
    debug(expected_files)
    assert total_files == {"B03": expected_files}
    assert site_metadata.shape == expected_shape
    assert np.allclose(site_metadata["x_micrometer"].unique(), x_mic_pos)
    assert np.allclose(site_metadata["y_micrometer"].unique(), y_mic_pos)
    assert np.allclose(site_metadata["pixel_size_z"], pixel_size_z)
    assert np.allclose(site_metadata["z_pixel"], z_pixel)
    assert np.allclose(site_metadata["pixel_size_x"], pixel_size_x)
    assert np.allclose(site_metadata["pixel_size_y"], pixel_size_y)
    assert np.allclose(site_metadata["x_pixel"], x_pixel)
    assert np.allclose(site_metadata["y_pixel"], y_pixel)
    assert np.allclose(site_metadata["bit_depth"], bit_depth)
    assert list(site_metadata["Time"]) == Time


def test_parse_yokogawa_metadata_multiwell():
    """
    This test checks two (incremental) valid behaviors:
        1) parse_yokogawa_metadata works for a mlf file spanning multiple wells
        2) When the images have "ambiguous" filenames (e.g. the plate part also
           includes a pattern that would match the well_id for another well),
           parse_yokogawa_metadata still works - and the per-well numbers of
           files are correct.
    """
    mlf_path_4 = (
        f"{path}MeasurementData_SingleWell2Sites_MultiZ"
        "_another_well_name_in_plate_name.mlf"
    )
    mrf_path_4 = f"{path}MeasurementDetail.mrf"

    site_metadata, file_numbers = parse_yokogawa_metadata(
        mrf_path_4, mlf_path_4
    )
    debug(file_numbers)
    assert file_numbers == {"C03": 4, "B03": 4, "D04": 8}


def test_manually_removing_overlap():
    import logging

    logging.warning(
        "test_manually_removing_overlap is relying on a wrong "
        "behavior, i.e. the detection of overlaps due to tol=0. "
        "To do: identify a DataFrame with an actual overlap and "
        "update this test."
    )
    # Tests the overlap detection and manually removing the overlaps
    site_metadata, _ = parse_yokogawa_metadata(mrf_path_1, mlf_path_1)
    site_metadata["x_micrometer_original"] = site_metadata["x_micrometer"]
    site_metadata["y_micrometer_original"] = site_metadata["y_micrometer"]
    overlapping_FOVs = run_overlap_check(site_metadata, tol=0)

    expected_overlaps = [{"B03": [2, 1]}]
    assert overlapping_FOVs == expected_overlaps
    site_metadata.loc[("B03", 2), "x_micrometer"] = -1032.2

    # Check that overlap has been successfully removed
    overlapping_FOVs_empty = run_overlap_check(site_metadata, tol=1e-10)
    assert len(overlapping_FOVs_empty) == 0

    # Load expected dataframe and set index + types correctly
    expected_site_metadata = pd.read_csv(
        Path(path) / "corrected_site_metadata_tiny_test.csv"
    )
    expected_site_metadata.set_index(["well_id", "FieldIndex"], inplace=True)
    expected_site_metadata["Time"] = pd.to_datetime(
        expected_site_metadata["Time"]
    )
    pd.testing.assert_frame_equal(site_metadata, expected_site_metadata)


def test_remove_overlap_when_sharing_corner(testdata_path: Path):
    csvfile = str(testdata_path / "metadata_files/metadata_issue_264.csv")
    site_metadata = pd.read_csv(csvfile)
    site_metadata.set_index(["well_id", "FieldIndex"], inplace=True)
    print(site_metadata.loc["A01"])
    site_metadata = remove_FOV_overlaps(site_metadata)
    print(site_metadata.loc["A01"])
    assert not site_metadata.isnull().values.any()
    overlapping_FOVs = run_overlap_check(site_metadata, tol=1e-10)
    assert not overlapping_FOVs

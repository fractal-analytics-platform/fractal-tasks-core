"""
Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
University of Zurich

Original authors:
Marco Franzon <marco.franzon@exact-lab.it>
Tommaso Comparin <tommaso.comparin@exact-lab.it>

This file is part of Fractal and was originally developed by eXact lab S.r.l.
<exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
Institute for Biomedical Research and Pelkmans Lab from the University of
Zurich.
"""
import pathlib

import numpy as np
import pandas as pd
from devtools import debug
from pytest import MonkeyPatch

from fractal_tasks_core.lib_regions_of_interest import prepare_FOV_ROI_table
from fractal_tasks_core.yokogawa_to_zarr import yokogawa_to_zarr

# FIXME: try with two channels

PIXEL_SIZE_X = 0.1625
PIXEL_SIZE_Y = 0.1625
PIXEL_SIZE_Z = 1.0

IMG_SIZE_X = 2560
IMG_SIZE_Y = 2160
NUM_Z_PLANES = 4

FOV_IDS = ["1", "2", "7", "9"]
FOV_NAMES = [f"FOV_{ID}" for ID in FOV_IDS]


def get_metadata_dataframe():
    """
    Create artificial metadata dataframe
    """
    df = pd.DataFrame(np.zeros((4, 11)), dtype=int)
    df.index = FOV_IDS
    df.columns = [
        "x_micrometer",
        "y_micrometer",
        "z_micrometer",
        "x_pixel",
        "y_pixel",
        "z_pixel",
        "pixel_size_x",
        "pixel_size_y",
        "pixel_size_z",
        "bit_depth",
        "time",
    ]
    img_size_x_micrometer = IMG_SIZE_X * PIXEL_SIZE_X
    img_size_y_micrometer = IMG_SIZE_Y * PIXEL_SIZE_Y
    df["x_micrometer"] = [
        0.0,
        img_size_x_micrometer,
        0.0,
        img_size_x_micrometer,
    ]
    df["y_micrometer"] = [
        0.0,
        0.0,
        img_size_y_micrometer,
        img_size_y_micrometer,
    ]
    df["z_micrometer"] = [0.0, 0.0, 0.0, 0.0]
    df["x_pixel"] = [IMG_SIZE_X] * 4
    df["y_pixel"] = [IMG_SIZE_Y] * 4
    df["z_pixel"] = [NUM_Z_PLANES] * 4
    df["pixel_size_x"] = [PIXEL_SIZE_X] * 4
    df["pixel_size_y"] = [PIXEL_SIZE_Y] * 4
    df["pixel_size_z"] = [PIXEL_SIZE_Z] * 4
    df["bit_depth"] = [16.0] * 4
    df["time"] = "2020-08-12 15:36:36.234000+0000"

    return df


images = [
    "plate_well_T0001F001L01A01Z01C01.png",
    "plate_well_T0001F002L01A01Z01C01.png",
    "plate_well_T0001F003L01A01Z01C01.png",
    "plate_well_T0001F004L01A01Z01C01.png",
    "plate_well_T0001F001L01A01Z02C01.png",
    "plate_well_T0001F002L01A01Z02C01.png",
    "plate_well_T0001F003L01A01Z02C01.png",
    "plate_well_T0001F004L01A01Z02C01.png",
    "plate_well_T0001F001L01A01Z03C01.png",
    "plate_well_T0001F002L01A01Z03C01.png",
    "plate_well_T0001F003L01A01Z03C01.png",
    "plate_well_T0001F004L01A01Z03C01.png",
    "plate_well_T0001F001L01A01Z04C01.png",
    "plate_well_T0001F002L01A01Z04C01.png",
    "plate_well_T0001F003L01A01Z04C01.png",
    "plate_well_T0001F004L01A01Z04C01.png",
]

chl_list = ["A01_C01"]
num_levels = 5
coarsening_factor_xy = 2
coarsening_factor_z = 1


def test_yokogawa_to_zarr(
    mocker,
    tmp_path: pathlib.Path,
    monkeypatch: MonkeyPatch,
):

    debug(tmp_path)

    # Mock list of images
    mocker.patch(
        "fractal_tasks_core.yokogawa_to_zarr.sorted", return_value=images
    )

    # Patch correct() function, to keep track of the number of calls
    logfile = (tmp_path / "log_function_correct.txt").resolve().as_posix()
    with open(logfile, "w") as log:
        log.write("")

    logfile = (tmp_path / "log_function_correct.txt").resolve().as_posix()

    def patched_imread(*args, **kwargs):
        with open(logfile, "a") as log:
            log.write("1\n")
        return np.ones((IMG_SIZE_Y, IMG_SIZE_X), dtype=np.uint16)

    def patched_read_zarr(*args):
        metadata_dataframe = get_metadata_dataframe()
        adata = prepare_FOV_ROI_table(metadata_dataframe)
        return adata

    monkeypatch.setattr(
        "fractal_tasks_core.yokogawa_to_zarr.read_zarr", patched_read_zarr
    )

    monkeypatch.setattr(
        "fractal_tasks_core.yokogawa_to_zarr.imread", patched_imread
    )

    monkeypatch.setattr(
        "fractal_tasks_core.yokogawa_to_zarr.extract_zyx_pixel_sizes",
        lambda x: [PIXEL_SIZE_Z, PIXEL_SIZE_Y, PIXEL_SIZE_X],
    )

    yokogawa_to_zarr(
        input_paths=[tmp_path / "*.png"],
        output_path=tmp_path,
        metadata=dict(
            channel_list=["A01_C01"],
            original_paths=["/tmp"],
            num_levels=5,
            coarsening_xy=2,
        ),
        component="row/column/fov/",
    )

    # Read number of calls to imread
    num_calls_imread = np.loadtxt(logfile, dtype=int).sum()
    # Subtract one for each channel, for the dummy call at the beginning of
    # the task (used to determine shape and dtype)
    num_calls_imread -= len(chl_list)

    assert num_calls_imread == len(images)

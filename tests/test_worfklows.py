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
import logging
from pathlib import Path

import pytest
from devtools import debug

from fractal_tasks_core.create_zarr_structure import create_zarr_structure
from fractal_tasks_core.illumination_correction import illumination_correction
from fractal_tasks_core.yokogawa_to_zarr import yokogawa_to_zarr


channel_parameters = {
    "A01_C01": {
        "label": "DAPI",
        "colormap": "00FFFF",
        "start": 0,
        "end": 700,
    },
    "A01_C02": {
        "label": "nanog",
        "colormap": "FF00FF",
        "start": 0,
        "end": 180,
    },
    "A02_C03": {
        "label": "Lamin B1",
        "colormap": "FFFF00",
        "start": 0,
        "end": 1500,
    },
}

num_levels = 2
coarsening_xy = 2


def test_workflow_yokogawa_to_zarr(tmp_path: Path, testdata_path: Path):

    # Init
    img_path = testdata_path / "png/*.png"
    zarr_path = tmp_path / "tmp_out/*.zarr"
    metadata = {}

    # Create zarr structure
    metadata_update = create_zarr_structure(
        input_paths=[img_path],
        output_path=zarr_path,
        channel_parameters=channel_parameters,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        metadata_table="mrf_mlf",
    )
    metadata.update(metadata_update)
    debug(metadata)

    # Yokogawa to zarr
    for component in metadata["well"]:
        yokogawa_to_zarr(
            input_paths=[zarr_path],
            output_path=zarr_path,
            rows=1,
            cols=2,
            metadata=metadata,
            component=component,
        )
    debug(metadata)


def test_workflow_illumination_correction(
    tmp_path: Path, testdata_path: Path, caplog: pytest.LogCaptureFixture
):

    # Setup caplog fixture, see
    # https://docs.pytest.org/en/stable/how-to/logging.html#caplog-fixture
    caplog.set_level(logging.INFO)

    # Init
    img_path = testdata_path / "png/*.png"
    zarr_path = tmp_path / "tmp_out/*.zarr"
    metadata = {}

    testdata_str = testdata_path.as_posix()
    illum_params = {
        "root_path_corr": f"{testdata_str}/illumination_correction/",
        "A01_C01": "illum_corr_matrix.png",
    }

    # Create zarr structure
    metadata_update = create_zarr_structure(
        input_paths=[img_path],
        output_path=zarr_path,
        channel_parameters=channel_parameters,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        metadata_table="mrf_mlf",
    )
    metadata.update(metadata_update)
    print(caplog.text)
    caplog.clear()

    # Yokogawa to zarr
    for component in metadata["well"]:
        yokogawa_to_zarr(
            input_paths=[zarr_path],
            output_path=zarr_path,
            rows=1,
            cols=2,
            metadata=metadata,
            component=component,
        )
    print(caplog.text)
    caplog.clear()

    # Illumination correction
    for component in metadata["well"]:
        illumination_correction(
            input_paths=[zarr_path],
            output_path=zarr_path,
            metadata=metadata,
            component=component,
            overwrite=True,
            dict_corr=illum_params,
        )
    print(caplog.text)
    caplog.clear()

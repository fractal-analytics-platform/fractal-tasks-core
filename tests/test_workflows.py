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
import shutil
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List

import pytest
from devtools import debug

from .utils import check_file_number
from .utils import validate_schema
from fractal_tasks_core.copy_ome_zarr import copy_ome_zarr
from fractal_tasks_core.create_ome_zarr import create_ome_zarr
from fractal_tasks_core.illumination_correction import illumination_correction
from fractal_tasks_core.maximum_intensity_projection import (
    maximum_intensity_projection,
)  # noqa
from fractal_tasks_core.yokogawa_to_ome_zarr import yokogawa_to_ome_zarr


allowed_channels = [
    {
        "label": "DAPI",
        "wavelength_id": "A01_C01",
        "colormap": "00FFFF",
        "start": 0,
        "end": 700,
    },
    {
        "wavelength_id": "A01_C02",
        "label": "nanog",
        "colormap": "FF00FF",
        "start": 0,
        "end": 180,
    },
    {
        "wavelength_id": "A02_C03",
        "label": "Lamin B1",
        "colormap": "FFFF00",
        "start": 0,
        "end": 1500,
    },
]

num_levels = 6
coarsening_xy = 2


@pytest.mark.xfail(reason="This would fail for a dataset with N>1 channels")
def test_create_ome_zarr_fail(tmp_path: Path, zenodo_images: str):

    allowed_channels = [
        {"label": "repeated label", "wavelength_id": "A01_C01"},
        {"label": "repeated label", "wavelength_id": "A01_C02"},
        {"label": "repeated label", "wavelength_id": "A02_C03"},
    ]

    # Init
    img_path = str(Path(zenodo_images) / "*.png")
    zarr_path = str(tmp_path / "tmp_out/*.zarr")

    # Create zarr structure
    with pytest.raises(ValueError):
        _ = create_ome_zarr(
            input_paths=[img_path],
            metadata={},
            output_path=zarr_path,
            allowed_channels=allowed_channels,
            num_levels=num_levels,
            coarsening_xy=coarsening_xy,
            metadata_table="mrf_mlf",
        )


metadata_inputs = ["use_mrf_mlf_files", "use_existing_csv_files"]


@pytest.mark.parametrize("metadata_input", metadata_inputs)
def test_yokogawa_to_ome_zarr(
    tmp_path: Path,
    zenodo_images: str,
    testdata_path: Path,
    metadata_input: str,
):

    # Select the kind of metadata_table input
    if metadata_input == "use_mrf_mlf_files":
        metadata_table = "mrf_mlf"
    if metadata_input == "use_existing_csv_files":
        testdata_str = testdata_path.as_posix()
        metadata_table = (
            f"{testdata_str}/metadata_files/"
            + "corrected_site_metadata_tiny_test.csv"
        )
    debug(metadata_table)

    # Init
    img_path = Path(zenodo_images) / "*.png"
    output_path = tmp_path / "output"

    # Create zarr structure
    metadata = {}
    metadata_update = create_ome_zarr(
        input_paths=[str(img_path)],
        output_path=str(output_path),
        metadata=metadata,
        allowed_channels=allowed_channels,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        metadata_table=metadata_table,
        image_extension="png",
    )
    metadata.update(metadata_update)
    debug(metadata)

    # Yokogawa to zarr
    for component in metadata["image"]:
        yokogawa_to_ome_zarr(
            input_paths=[str(output_path)],
            output_path=str(output_path),
            metadata=metadata,
            component=component,
        )
    debug(metadata)

    # OME-NGFF JSON validation
    image_zarr = Path(output_path / metadata["image"][0])
    well_zarr = image_zarr.parent
    plate_zarr = image_zarr.parents[2]
    validate_schema(path=str(image_zarr), type="image")
    validate_schema(path=str(well_zarr), type="well")
    validate_schema(path=str(plate_zarr), type="plate")

    check_file_number(zarr_path=image_zarr)


@pytest.mark.skip(reason="ongoing refactor - see issue #300")
def test_MIP(
    tmp_path: Path,
    zenodo_zarr: List[str],
    zenodo_zarr_metadata: List[Dict[str, Any]],
):

    # Init
    zarr_path = tmp_path / "tmp_out/*.zarr"
    zarr_path_mip = tmp_path / "tmp_out_mip/*.zarr"

    # Load zarr array from zenodo
    zenodo_zarr_3D, zenodo_zarr_2D = zenodo_zarr[:]
    metadata_3D, metadata_2D = zenodo_zarr_metadata[:]
    shutil.copytree(
        zenodo_zarr_3D, str(zarr_path.parent / Path(zenodo_zarr_3D).name)
    )
    metadata = metadata_3D.copy()

    # Replicate
    metadata_update = copy_ome_zarr(
        input_paths=[str(zarr_path)],
        output_path=str(zarr_path_mip),
        metadata=metadata,
        project_to_2D=True,
        suffix="mip",
    )
    metadata.update(metadata_update)
    debug(metadata)

    # MIP
    for component in metadata["image"]:
        maximum_intensity_projection(
            input_paths=[zarr_path_mip],
            output_path=zarr_path_mip,
            metadata=metadata,
            component=component,
        )

    # OME-NGFF JSON validation
    image_zarr = Path(zarr_path_mip.parent / metadata["image"][0])
    debug(image_zarr)
    well_zarr = image_zarr.parent
    plate_zarr = image_zarr.parents[2]
    validate_schema(path=str(image_zarr), type="image")
    validate_schema(path=str(well_zarr), type="well")
    validate_schema(path=str(plate_zarr), type="plate")


@pytest.mark.skip(reason="ongoing refactor - see issue #300")
def test_illumination_correction(
    tmp_path: Path,
    testdata_path: Path,
    zenodo_images: str,
    caplog: pytest.LogCaptureFixture,
):

    # Setup caplog fixture, see
    # https://docs.pytest.org/en/stable/how-to/logging.html#caplog-fixture
    caplog.set_level(logging.INFO)

    # Init
    img_path = Path(zenodo_images) / "*.png"
    zarr_path = tmp_path / "tmp_out/*.zarr"
    metadata = {}

    testdata_str = testdata_path.as_posix()
    illum_params = {
        "root_path_corr": f"{testdata_str}/illumination_correction/",
        "A01_C01": "illum_corr_matrix.png",
    }

    # Create zarr structure
    metadata_update = create_ome_zarr(
        input_paths=[str(img_path)],
        output_path=str(zarr_path),
        metadata=metadata,
        allowed_channels=allowed_channels,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        metadata_table="mrf_mlf",
    )
    metadata.update(metadata_update)
    print(caplog.text)
    caplog.clear()

    # Yokogawa to zarr
    for component in metadata["image"]:
        yokogawa_to_ome_zarr(
            input_paths=[str(zarr_path)],
            output_path=str(zarr_path),
            metadata=metadata,
            component=component,
        )
    print(caplog.text)
    caplog.clear()

    # Illumination correction
    for component in metadata["image"]:
        illumination_correction(
            input_paths=[str(zarr_path)],
            output_path=str(zarr_path),
            metadata=metadata,
            component=component,
            overwrite=True,
            dict_corr=illum_params,
        )
    print(caplog.text)
    caplog.clear()

    # OME-NGFF JSON validation
    image_zarr = Path(zarr_path.parent / metadata["image"][0])
    well_zarr = image_zarr.parent
    plate_zarr = image_zarr.parents[2]
    validate_schema(path=str(image_zarr), type="image")
    validate_schema(path=str(well_zarr), type="well")
    validate_schema(path=str(plate_zarr), type="plate")

    check_file_number(zarr_path=image_zarr)

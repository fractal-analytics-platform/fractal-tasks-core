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
from pathlib import Path
from typing import Sequence

import pytest
from devtools import debug

from .utils import check_file_number
from .utils import validate_schema
from fractal_tasks_core.copy_ome_zarr import (
    copy_ome_zarr,
)  # noqa
from fractal_tasks_core.create_ome_zarr_multiplex import (
    create_ome_zarr_multiplex,
)
from fractal_tasks_core.maximum_intensity_projection import (
    maximum_intensity_projection,
)  # noqa
from fractal_tasks_core.yokogawa_to_ome_zarr import yokogawa_to_ome_zarr


single_cycle_allowed_channels_no_label = [
    {
        "wavelength_id": "A01_C01",
        "colormap": "00FFFF",
        "start": 0,
        "end": 700,
    },
    {
        "wavelength_id": "A01_C02",
        "colormap": "FF00FF",
        "start": 0,
        "end": 180,
    },
    {
        "wavelength_id": "A02_C03",
        "colormap": "FFFF00",
        "start": 0,
        "end": 1500,
    },
]

allowed_channels = {
    "0": single_cycle_allowed_channels_no_label,
    "1": single_cycle_allowed_channels_no_label,
}

num_levels = 6
coarsening_xy = 2


def test_multiplexing_create_ome_zarr_fail(
    tmp_path: Path, zenodo_images_multiplex: Sequence[str]
):

    single_cycle_allowed_channels = [
        {"wavelength_id": "A01_C01", "label": "my label"}
    ]
    allowed_channels = {
        "0": single_cycle_allowed_channels,
        "1": single_cycle_allowed_channels,
    }

    # Init
    img_paths = [
        str(Path(cycle_folder) / "*.png")
        for cycle_folder in zenodo_images_multiplex
    ]
    zarr_path = tmp_path / "tmp_out/"

    # Create zarr structure
    debug(img_paths)
    with pytest.raises(ValueError):
        _ = create_ome_zarr_multiplex(
            input_paths=img_paths,
            output_path=str(zarr_path),
            metadata={},
            allowed_channels=allowed_channels,
            num_levels=num_levels,
            coarsening_xy=coarsening_xy,
            metadata_table="mrf_mlf",
        )


metadata_inputs = ["use_mrf_mlf_files", "use_existing_csv_files"]


@pytest.mark.parametrize("metadata_input", metadata_inputs)
def test_multiplexing_yokogawa_to_ome_zarr(
    tmp_path: Path,
    zenodo_images_multiplex: Sequence[str],
    metadata_input: str,
    testdata_path: Path,
):

    # Select the kind of metadata_table input
    if metadata_input == "use_mrf_mlf_files":
        metadata_table = "mrf_mlf"
    if metadata_input == "use_existing_csv_files":
        testdata_str = testdata_path.as_posix()
        metadata_table = {
            "0": f"{testdata_str}/metadata_files/"
            "corrected_site_metadata_tiny_test.csv",
            "1": f"{testdata_str}/metadata_files/"
            "corrected_site_metadata_tiny_test.csv",
        }

    debug(metadata_table)

    # Init
    zarr_path = tmp_path / "tmp_out/"
    metadata = {}

    # Create zarr structure
    metadata_update = create_ome_zarr_multiplex(
        input_paths=zenodo_images_multiplex,
        output_path=str(zarr_path),
        metadata=metadata,
        image_extension="png",
        allowed_channels=allowed_channels,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        metadata_table=metadata_table,
    )
    metadata.update(metadata_update)
    debug(metadata)

    # Yokogawa to zarr
    for component in metadata["image"]:
        yokogawa_to_ome_zarr(
            input_paths=[str(zarr_path)],
            output_path=str(zarr_path),
            metadata=metadata,
            component=component,
        )
    debug(metadata)

    # OME-NGFF JSON validation
    image_zarr_0 = zarr_path / metadata["image"][0]
    image_zarr_1 = zarr_path / metadata["image"][1]
    well_zarr = image_zarr_0.parent
    plate_zarr = image_zarr_0.parents[2]
    validate_schema(path=str(image_zarr_0), type="image")
    validate_schema(path=str(image_zarr_1), type="image")
    validate_schema(path=str(well_zarr), type="well")
    validate_schema(path=str(plate_zarr), type="plate")

    check_file_number(zarr_path=image_zarr_0)
    check_file_number(zarr_path=image_zarr_1)


def test_multiplexing_MIP(
    tmp_path: Path, zenodo_images_multiplex: Sequence[str]
):

    # Init
    zarr_path = tmp_path / "tmp_out/"
    zarr_path_mip = tmp_path / "tmp_out_mip/"
    metadata = {}

    # Create zarr structure
    metadata_update = create_ome_zarr_multiplex(
        input_paths=zenodo_images_multiplex,
        output_path=str(zarr_path),
        metadata=metadata,
        allowed_channels=allowed_channels,
        image_extension="png",
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        metadata_table="mrf_mlf",
    )
    metadata.update(metadata_update)
    debug(metadata)

    # Yokogawa to zarr
    for component in metadata["image"]:
        yokogawa_to_ome_zarr(
            input_paths=[str(zarr_path)],
            output_path=str(zarr_path),
            metadata=metadata,
            component=component,
        )
    debug(metadata)

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
            input_paths=[str(zarr_path_mip)],
            output_path=str(zarr_path_mip),
            metadata=metadata,
            component=component,
        )

    # OME-NGFF JSON validation
    image_zarr_0 = zarr_path_mip / metadata["image"][0]
    image_zarr_1 = zarr_path_mip / metadata["image"][1]
    well_zarr = image_zarr_0.parent
    plate_zarr = image_zarr_0.parents[2]
    validate_schema(path=str(image_zarr_0), type="image")
    validate_schema(path=str(image_zarr_1), type="image")
    validate_schema(path=str(well_zarr), type="well")
    validate_schema(path=str(plate_zarr), type="plate")

    check_file_number(zarr_path=image_zarr_0)
    check_file_number(zarr_path=image_zarr_1)

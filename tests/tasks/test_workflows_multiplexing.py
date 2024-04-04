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

from ._validation import check_file_number
from ._validation import validate_schema
from fractal_tasks_core.tasks.cellvoyager_to_ome_zarr_compute import (
    cellvoyager_to_ome_zarr_compute,
)
from fractal_tasks_core.tasks.cellvoyager_to_ome_zarr_init_multiplex import (
    cellvoyager_to_ome_zarr_init_multiplex,
)
from fractal_tasks_core.tasks.copy_ome_zarr_hcs_plate import (
    copy_ome_zarr_hcs_plate,
)
from fractal_tasks_core.tasks.io_models import MultiplexingAcquisition
from fractal_tasks_core.tasks.maximum_intensity_projection import (
    maximum_intensity_projection,
)
from fractal_tasks_core.zarr_utils import OverwriteNotAllowedError


single_cycle_allowed_channels_no_label = [
    {
        "wavelength_id": "A01_C01",
        "color": "00FFFF",
        "window": {"start": 0, "end": 700},
    },
    {
        "wavelength_id": "A01_C02",
        "color": "FF00FF",
        "window": {"start": 0, "end": 180},
    },
    {
        "wavelength_id": "A02_C03",
        "color": "FFFF00",
        "window": {"start": 0, "end": 1500},
    },
]

num_levels = 6
coarsening_xy = 2


def test_multiplexing_create_ome_zarr_fail(
    tmp_path: Path, zenodo_images_multiplex: Sequence[str]
):
    single_cycle_allowed_channels = [
        {"wavelength_id": "A01_C01", "label": "my label"}
    ]
    acquisitions = {
        "0": MultiplexingAcquisition(
            image_dir=zenodo_images_multiplex[0],
            allowed_channels=single_cycle_allowed_channels,
        ),
        "1": MultiplexingAcquisition(
            image_dir=zenodo_images_multiplex[1],
            allowed_channels=single_cycle_allowed_channels,
        ),
    }

    # Init
    zarr_dir = tmp_path / "tmp_out/"

    # Create zarr structure
    debug(zenodo_images_multiplex)
    with pytest.raises(ValueError):
        _ = cellvoyager_to_ome_zarr_init_multiplex(
            zarr_urls=[],
            zarr_dir=zarr_dir,
            acquisitions=acquisitions,
            num_levels=num_levels,
            coarsening_xy=coarsening_xy,
            metadata_table_files=None,
        )


metadata_inputs = ["use_mrf_mlf_files", "use_existing_csv_files"]


@pytest.mark.parametrize("metadata_input", metadata_inputs)
def test_multiplexing_compute(
    tmp_path: Path,
    zenodo_images_multiplex: Sequence[str],
    metadata_input: str,
    testdata_path: Path,
):

    # Select the kind of metadata_table_files input
    if metadata_input == "use_mrf_mlf_files":
        metadata_table_files = None
    if metadata_input == "use_existing_csv_files":
        testdata_str = testdata_path.as_posix()
        metadata_table_files = {
            "0": f"{testdata_str}/metadata_files/"
            "corrected_site_metadata_tiny_test.csv",
            "1": f"{testdata_str}/metadata_files/"
            "corrected_site_metadata_tiny_test.csv",
        }

    debug(metadata_table_files)

    acquisitions = {
        "0": MultiplexingAcquisition(
            image_dir=zenodo_images_multiplex[0],
            allowed_channels=single_cycle_allowed_channels_no_label,
        ),
        "1": MultiplexingAcquisition(
            image_dir=zenodo_images_multiplex[1],
            allowed_channels=single_cycle_allowed_channels_no_label,
        ),
    }

    # Init
    zarr_dir = str(tmp_path / "tmp_out/")

    # Create zarr structure
    parallelization_list = cellvoyager_to_ome_zarr_init_multiplex(
        zarr_urls=[],
        zarr_dir=zarr_dir,
        acquisitions=acquisitions,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        image_extension="png",
        metadata_table_files=metadata_table_files,
    )

    debug(parallelization_list)

    # Run again, with overwrite=True
    parallelization_list_2 = cellvoyager_to_ome_zarr_init_multiplex(
        zarr_urls=[],
        zarr_dir=zarr_dir,
        acquisitions=acquisitions,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        image_extension="png",
        metadata_table_files=metadata_table_files,
        overwrite=True,
    )
    assert parallelization_list_2 == parallelization_list

    # Run again, with overwrite=False
    with pytest.raises(OverwriteNotAllowedError):
        _ = cellvoyager_to_ome_zarr_init_multiplex(
            zarr_urls=[],
            zarr_dir=zarr_dir,
            acquisitions=acquisitions,
            num_levels=num_levels,
            coarsening_xy=coarsening_xy,
            image_extension="png",
            metadata_table_files=metadata_table_files,
            overwrite=False,
        )

    # Convert to OME-Zarr
    image_list_updates = []
    for image in parallelization_list:
        image_list_updates += cellvoyager_to_ome_zarr_compute(
            zarr_url=image["zarr_url"],
            init_args=image["init_args"],
        )["image_list_updates"]
    debug(image_list_updates)

    # Check image_list_updates
    expected_image_list_update = [
        {
            "zarr_url": (
                f"{zarr_dir}/20200812-CardiomyocyteDifferentiation14"
                "-Cycle1.zarr/B/03/0/"
            ),
            "attributes": {
                "plate": "20200812-CardiomyocyteDifferentiation14-Cycle1.zarr",
                "well": "B03",
                "acquisition": 0,
            },
            "types": {
                "is_3D": True,
            },
        },
        {
            "zarr_url": (
                f"{zarr_dir}/20200812-CardiomyocyteDifferentiation14"
                "-Cycle1.zarr/B/03/1/"
            ),
            "attributes": {
                "plate": "20200812-CardiomyocyteDifferentiation14-Cycle1.zarr",
                "well": "B03",
                "acquisition": 1,
            },
            "types": {
                "is_3D": True,
            },
        },
    ]
    assert image_list_updates == expected_image_list_update

    # OME-NGFF JSON validation
    image_zarr_0 = Path(zarr_dir) / parallelization_list[0]["zarr_url"]
    image_zarr_1 = Path(zarr_dir) / parallelization_list[1]["zarr_url"]
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
    zarr_dir = tmp_path / "tmp_out/"

    acquisitions = {
        "0": MultiplexingAcquisition(
            image_dir=zenodo_images_multiplex[0],
            allowed_channels=single_cycle_allowed_channels_no_label,
        ),
        "1": MultiplexingAcquisition(
            image_dir=zenodo_images_multiplex[1],
            allowed_channels=single_cycle_allowed_channels_no_label,
        ),
    }

    # Create zarr structure
    parallelization_list = cellvoyager_to_ome_zarr_init_multiplex(
        zarr_urls=[],
        zarr_dir=str(zarr_dir),
        acquisitions=acquisitions,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        image_extension="png",
        metadata_table_files=None,
    )
    debug(parallelization_list)

    # Convert to OME-Zarr
    image_list_updates = []
    for image in parallelization_list:
        image_list_updates += cellvoyager_to_ome_zarr_compute(
            zarr_url=image["zarr_url"],
            init_args=image["init_args"],
        )["image_list_updates"]
    debug(image_list_updates)

    zarr_urls = []
    for image in image_list_updates:
        zarr_urls.append(image["zarr_url"])

    # Replicate
    parallelization_list = copy_ome_zarr_hcs_plate(
        zarr_urls=zarr_urls,
        zarr_dir=str(zarr_dir),
        overwrite=True,
    )
    debug(parallelization_list)

    # MIP
    image_list_updates = []
    for image in parallelization_list:
        image_list_updates += maximum_intensity_projection(
            zarr_url=image["zarr_url"],
            init_args=image["init_args"],
            overwrite=True,
        )["image_list_updates"]

    # OME-NGFF JSON validation
    image_zarr_0 = Path(zarr_dir) / parallelization_list[0]["zarr_url"]
    image_zarr_1 = Path(zarr_dir) / parallelization_list[1]["zarr_url"]
    well_zarr = image_zarr_0.parent
    plate_zarr = image_zarr_0.parents[2]
    validate_schema(path=str(image_zarr_0), type="image")
    validate_schema(path=str(image_zarr_1), type="image")
    validate_schema(path=str(well_zarr), type="well")
    validate_schema(path=str(plate_zarr), type="plate")

    check_file_number(zarr_path=image_zarr_0)
    check_file_number(zarr_path=image_zarr_1)

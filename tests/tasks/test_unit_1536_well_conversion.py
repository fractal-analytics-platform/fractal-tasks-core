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
from pathlib import Path

from devtools import debug

from ._validation import validate_schema
from fractal_tasks_core.cellvoyager.metadata import parse_yokogawa_metadata
from fractal_tasks_core.channels import OmeroChannel
from fractal_tasks_core.channels import Window
from fractal_tasks_core.tasks.cellvoyager_to_ome_zarr_compute import (
    cellvoyager_to_ome_zarr_compute,
)
from fractal_tasks_core.tasks.cellvoyager_to_ome_zarr_init import (
    cellvoyager_to_ome_zarr_init,
)
from fractal_tasks_core.tasks.cellvoyager_to_ome_zarr_init_multiplex import (
    cellvoyager_to_ome_zarr_init_multiplex,
)
from fractal_tasks_core.tasks.io_models import MultiplexingAcquisition


def test_1536_well_metadata_conversion(syn_1536_images: str):
    mlf_path = Path(syn_1536_images) / "MeasurementData.mlf"
    mrf_path = Path(syn_1536_images) / "MeasurementDetail.mrf"
    site_metadata, number_of_files = parse_yokogawa_metadata(
        mrf_path, mlf_path
    )
    assert number_of_files == {"B03.a1": 2}
    assert len(site_metadata) == 2


def test_1536_ome_zarr_conversion(tmp_path: Path, syn_1536_images: str):
    allowed_channels = [
        OmeroChannel(
            label="Channel 1",
            wavelength_id="A01_C01",
            color="00FFFF",
            window=Window(start=0, end=5000),
        )
    ]
    num_levels = 2
    coarsening_xy = 2
    zarr_dir = str(tmp_path / "tmp_syn_out_1536/")

    # Create zarr structure
    parallelization_list = cellvoyager_to_ome_zarr_init(
        zarr_dir=zarr_dir,
        image_dirs=[syn_1536_images],
        allowed_channels=allowed_channels,
        image_extension="png",
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        overwrite=True,
    )
    debug(parallelization_list)

    image_list_updates = []
    # Yokogawa to zarr
    for image in parallelization_list["parallelization_list"]:
        image_list_updates += cellvoyager_to_ome_zarr_compute(
            zarr_url=image["zarr_url"],
            init_args=image["init_args"],
        )["image_list_updates"]
    debug(image_list_updates)

    # OME-NGFF JSON validation
    image_zarr = Path(
        parallelization_list["parallelization_list"][0]["zarr_url"]
    )
    well_zarr = image_zarr.parent
    plate_zarr = image_zarr.parents[2]

    validate_schema(path=str(image_zarr), type="image")
    validate_schema(path=str(well_zarr), type="well")
    validate_schema(path=str(plate_zarr), type="plate")


def test_1536_multiplexing_ome_zarr_conversion(
    tmp_path: Path, syn_1536_images: str
):
    acquisition_1 = MultiplexingAcquisition(
        image_dir=syn_1536_images,
        allowed_channels=[
            OmeroChannel(
                label="Channel 1",
                wavelength_id="A01_C01",
                color="00FFFF",
                window=Window(start=0, end=5000),
            )
        ],
    )
    acquisition_2 = MultiplexingAcquisition(
        image_dir=syn_1536_images,
        allowed_channels=[
            OmeroChannel(
                label="Channel 2",
                wavelength_id="A01_C01",
                color="00FFFF",
                window=Window(start=0, end=5000),
            )
        ],
    )
    num_levels = 2
    coarsening_xy = 2
    zarr_dir = str(tmp_path / "tmp_syn_multiplex_out_1536/")

    # Create zarr structure
    parallelization_list = cellvoyager_to_ome_zarr_init_multiplex(
        zarr_dir=zarr_dir,
        acquisitions={"0": acquisition_1, "1": acquisition_2},
        image_extension="png",
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        overwrite=True,
    )
    debug(parallelization_list)

    image_list_updates = []
    # Yokogawa to zarr
    for image in parallelization_list["parallelization_list"]:
        image_list_updates += cellvoyager_to_ome_zarr_compute(
            zarr_url=image["zarr_url"],
            init_args=image["init_args"],
        )["image_list_updates"]
    debug(image_list_updates)

    # OME-NGFF JSON validation
    image_zarr = Path(
        parallelization_list["parallelization_list"][0]["zarr_url"]
    )
    well_zarr = image_zarr.parent
    plate_zarr = image_zarr.parents[2]

    validate_schema(path=str(image_zarr), type="image")
    validate_schema(path=str(well_zarr), type="well")
    validate_schema(path=str(plate_zarr), type="plate")

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

from devtools import debug

from .utils import check_file_number
from .utils import validate_schema
from fractal_tasks_core.create_zarr_structure_multiplex import (
    create_zarr_structure_multiplex,
)
from fractal_tasks_core.maximum_intensity_projection import (
    maximum_intensity_projection,
)  # noqa
from fractal_tasks_core.replicate_zarr_structure import (
    replicate_zarr_structure,
)  # noqa
from fractal_tasks_core.yokogawa_to_zarr import yokogawa_to_zarr


single_cycle_channel_parameters = {
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
channel_parameters = {
    "0": single_cycle_channel_parameters,
    "1": single_cycle_channel_parameters,
}

num_levels = 6
coarsening_xy = 2


def test_workflow_multiplexing(
    tmp_path: Path, zenodo_images_multiplex: Sequence[Path]
):

    # Init
    img_paths = [
        cycle_folder / "*.png" for cycle_folder in zenodo_images_multiplex
    ]
    zarr_path = tmp_path / "tmp_out/*.zarr"
    metadata = {}

    # Create zarr structure
    debug(img_paths)
    metadata_update = create_zarr_structure_multiplex(
        input_paths=img_paths,
        output_path=zarr_path,
        channel_parameters=channel_parameters,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        metadata_table="mrf_mlf",
    )
    metadata.update(metadata_update)
    debug(metadata)

    # Yokogawa to zarr
    for component in metadata["image"]:
        yokogawa_to_zarr(
            input_paths=[zarr_path],
            output_path=zarr_path,
            metadata=metadata,
            component=component,
        )
    debug(metadata)

    # OME-NGFF JSON validation
    image_zarr_0 = Path(zarr_path.parent / metadata["image"][0])
    image_zarr_1 = Path(zarr_path.parent / metadata["image"][1])
    well_zarr = image_zarr_0.parent
    plate_zarr = image_zarr_0.parents[2]
    validate_schema(path=str(image_zarr_0), type="image")
    validate_schema(path=str(image_zarr_1), type="image")
    validate_schema(path=str(well_zarr), type="well")
    validate_schema(path=str(plate_zarr), type="plate")

    check_file_number(zarr_path=image_zarr_0)
    check_file_number(zarr_path=image_zarr_1)


def test_workflow_multiplexing_MIP(
    tmp_path: Path, zenodo_images_multiplex: Sequence[Path]
):

    # Init
    img_paths = [
        cycle_folder / "*.png" for cycle_folder in zenodo_images_multiplex
    ]
    zarr_path = tmp_path / "tmp_out/*.zarr"
    zarr_path_mip = tmp_path / "tmp_out_mip/*.zarr"
    metadata = {}

    # Create zarr structure
    debug(img_paths)
    metadata_update = create_zarr_structure_multiplex(
        input_paths=img_paths,
        output_path=zarr_path,
        channel_parameters=channel_parameters,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        metadata_table="mrf_mlf",
    )
    metadata.update(metadata_update)
    debug(metadata)

    # Yokogawa to zarr
    for component in metadata["image"]:
        yokogawa_to_zarr(
            input_paths=[zarr_path],
            output_path=zarr_path,
            metadata=metadata,
            component=component,
        )
    debug(metadata)

    # Replicate
    metadata_update = replicate_zarr_structure(
        input_paths=[zarr_path],
        output_path=zarr_path_mip,
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
    image_zarr_0 = Path(zarr_path_mip.parent / metadata["image"][0])
    image_zarr_1 = Path(zarr_path_mip.parent / metadata["image"][1])
    well_zarr = image_zarr_0.parent
    plate_zarr = image_zarr_0.parents[2]
    validate_schema(path=str(image_zarr_0), type="image")
    validate_schema(path=str(image_zarr_1), type="image")
    validate_schema(path=str(well_zarr), type="well")
    validate_schema(path=str(plate_zarr), type="plate")

    check_file_number(zarr_path=image_zarr_0)
    check_file_number(zarr_path=image_zarr_1)

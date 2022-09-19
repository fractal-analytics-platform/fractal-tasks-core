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
import json
import urllib.request
from pathlib import Path

from devtools import debug
from jsonschema import validate

import fractal_tasks_core
from fractal_tasks_core.create_zarr_structure import create_zarr_structure
from fractal_tasks_core.yokogawa_to_zarr import yokogawa_to_zarr


__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__

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

    debug(zarr_path)

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
            metadata=metadata,
            component=component,
        )
    debug(metadata)

    # Validate final JSON
    url_plate = (
        "https://raw.githubusercontent.com/ome/ngff/main/"
        f"{__OME_NGFF_VERSION__}/"
        "schemas/plate.schema"
    )
    url_well = (
        "https://raw.githubusercontent.com/ome/ngff/main/"
        f"{__OME_NGFF_VERSION__}/"
        "schemas/well.schema"
    )
    url_image = (
        "https://raw.githubusercontent.com/ome/ngff/main/"
        f"{__OME_NGFF_VERSION__}/"
        "schemas/image.schema"
    )
    debug(url_plate)
    debug(url_well)
    debug(url_image)
    with urllib.request.urlopen(url_plate) as url:
        plate_schema = json.load(url)
    with urllib.request.urlopen(url_well) as url:
        well_schema = json.load(url)
    with urllib.request.urlopen(url_image) as url:
        image_schema = json.load(url)
    print(plate_schema)

    image_zarr = Path(zarr_path.parent / metadata["well"][0])
    well_zarr = image_zarr.parent
    plate_zarr = image_zarr.parents[2]
    debug(image_zarr)
    debug(well_zarr)
    debug(plate_zarr)

    with open(f"{plate_zarr}/.zattrs", "r") as fin:
        plate_zattrs = json.load(fin)
    with open(f"{well_zarr}/.zattrs", "r") as fin:
        well_zattrs = json.load(fin)
    with open(f"{image_zarr}/.zattrs", "r") as fin:
        image_zattrs = json.load(fin)

    validate(instance=plate_zattrs, schema=plate_schema)
    validate(instance=well_zattrs, schema=well_schema)
    validate(instance=image_zattrs, schema=image_schema)

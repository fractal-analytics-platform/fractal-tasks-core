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
import os
from pathlib import Path

from devtools import debug

from fractal_tasks_core.ome_zarr.channels import OmeroChannel
from fractal_tasks_core.tasks.create_ome_zarr import create_ome_zarr
from fractal_tasks_core.tasks.yokogawa_to_ome_zarr import yokogawa_to_ome_zarr


allowed_channels = [
    OmeroChannel(wavelength_id="A01_C01"),
    OmeroChannel(wavelength_id="A01_C02"),
    OmeroChannel(wavelength_id="A02_C03"),
]
num_levels = 2
coarsening_xy = 2


# Init
img_path = "../images/10.5281_zenodo.7059515/"
if not os.path.isdir(Path(img_path).parent):
    raise FileNotFoundError(
        f"{Path(img_path).parent} is missing,"
        " try running ./fetch_test_data_from_zenodo.sh"
    )
zarr_path = "tmp_out/"
metadata: dict = {}

# Create zarr structure
metadata_update = create_ome_zarr(
    input_paths=[img_path],
    output_path=zarr_path,
    allowed_channels=allowed_channels,
    image_extension="png",
    metadata=metadata,
    num_levels=num_levels,
    coarsening_xy=coarsening_xy,
)
metadata.update(metadata_update)
debug(metadata)

# Yokogawa to zarr
for component in metadata["image"]:
    yokogawa_to_ome_zarr(
        input_paths=[zarr_path],
        output_path=zarr_path,
        metadata=metadata,
        component=component,
    )
debug(metadata)

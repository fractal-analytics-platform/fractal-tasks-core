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
from fractal_tasks_core.ome_zarr.channels import Window
from fractal_tasks_core.tasks.create_ome_zarr import create_ome_zarr
from fractal_tasks_core.tasks.illumination_correction import (
    illumination_correction,
)
from fractal_tasks_core.tasks.yokogawa_to_ome_zarr import yokogawa_to_ome_zarr


allowed_channels = [
    OmeroChannel(
        label="DAPI",
        wavelength_id="A01_C01",
        color="00FFFF",
        window=Window(start=0, end=700),
    ),
    OmeroChannel(
        wavelength_id="A01_C02",
        label="nanog",
        color="FF00FF",
        window=Window(start=0, end=180),
    ),
    OmeroChannel(
        wavelength_id="A02_C03",
        label="Lamin B1",
        color="FFFF00",
        window=Window(start=0, end=1500),
    ),
]

num_levels = 4
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

# Illumination correction
cwd = Path(__file__).parent.resolve().as_posix()
dict_corr = {
    "A01_C01": "illum_corr_matrix.png",
    "A01_C02": "illum_corr_matrix.png",
    "A02_C03": "illum_corr_matrix.png",
}
for component in metadata["image"]:
    illumination_correction(
        input_paths=[zarr_path],
        output_path=zarr_path,
        metadata=metadata,
        component=component,
        overwrite=True,
        illumination_profiles_folder=f"{cwd}/parameters",
        dict_corr=dict_corr,
        background=100,
    )
debug(metadata)

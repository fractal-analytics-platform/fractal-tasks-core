"""
Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
University of Zurich

Original authors:
Marco Franzon <marco.franzon@exact-lab.it>
Tommaso Comparin <tommaso.comparin@exact-lab.it>
Joel Lüthi <joel.luethi@uzh.ch>

This file is part of Fractal and was originally developed by eXact lab S.r.l.
<exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
Institute for Biomedical Research and Pelkmans Lab from the University of
Zurich.
"""
import os
from pathlib import Path

from devtools import debug

from fractal_tasks_core.channels import OmeroChannel
from fractal_tasks_core.channels import Window
from fractal_tasks_core.tasks.cellvoyager_to_ome_zarr_compute import (
    cellvoyager_to_ome_zarr_compute,
)
from fractal_tasks_core.tasks.cellvoyager_to_ome_zarr_init import (
    cellvoyager_to_ome_zarr_init,
)
from fractal_tasks_core.tasks.copy_ome_zarr_hcs_plate import (
    copy_ome_zarr_hcs_plate,
)
from fractal_tasks_core.tasks.projection import (
    projection,
)

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
num_levels = 2
coarsening_xy = 2


# Init
img_path = "../images/10.5281_zenodo.7059515/"
if not os.path.isdir(Path(img_path).parent):
    raise FileNotFoundError(
        f"{Path(img_path).parent} is missing,"
        " try running ./fetch_test_data_from_zenodo.sh"
    )
zarr_dir = "tmp_out"

# Create zarr structure
parallelization_list = cellvoyager_to_ome_zarr_init(
    zarr_dir=zarr_dir,
    image_dirs=[img_path],
    allowed_channels=allowed_channels,
    image_extension="png",
    num_levels=num_levels,
    coarsening_xy=coarsening_xy,
    overwrite=True,
)
debug(parallelization_list)

image_list_updates = []
# Convert image to Zarr
for image in parallelization_list["parallelization_list"]:
    image_list_updates += cellvoyager_to_ome_zarr_compute(
        zarr_url=image["zarr_url"],
        init_args=image["init_args"],
    )["image_list_updates"]
debug(image_list_updates)

zarr_urls = [
    "tmp_out/20200812-CardiomyocyteDifferentiation14-Cycle1.zarr/B/03/0/"
]

# Create an MIP
parallelization_list = copy_ome_zarr_hcs_plate(
    zarr_urls=zarr_urls, zarr_dir="tmp_out", overwrite=True
)
debug(parallelization_list)

for image in parallelization_list["parallelization_list"]:
    projection(
        zarr_url=image["zarr_url"],
        init_args=image["init_args"],
        # overwrite=True,
    )

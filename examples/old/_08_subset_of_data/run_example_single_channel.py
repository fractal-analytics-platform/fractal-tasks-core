import os

from devtools import debug

from fractal_tasks_core.channels import OmeroChannel
from fractal_tasks_core.channels import Window
from fractal_tasks_core.tasks.copy_ome_zarr import copy_ome_zarr
from fractal_tasks_core.tasks.create_ome_zarr import create_ome_zarr
from fractal_tasks_core.tasks.projection import (
    projection,
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


num_levels = 6
coarsening_xy = 2


# Init
img_path = "../images/10.5281_zenodo.7057076/"
if not os.path.isdir(img_path):
    raise FileNotFoundError(
        f"{img_path} is missing,"
        " try running ./fetch_test_data_from_zenodo.sh"
    )
zarr_path = "tmp_out"
metadata: dict = {}

# Create zarr structure
metadata_update = create_ome_zarr(
    input_paths=[img_path],
    output_path=zarr_path,
    metadata=metadata,
    image_extension="png",
    image_glob_patterns=["*A02*C03*"],
    allowed_channels=allowed_channels,
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

# Copy zarr structure
metadata_update = copy_ome_zarr(
    input_paths=[zarr_path],
    output_path=zarr_path,
    metadata=metadata,
    suffix="mip",
)
metadata.update(metadata_update)
debug(metadata)

# Make MIP
for component in metadata["image"]:
    metadata_update = projection(
        input_paths=[zarr_path],
        output_path=zarr_path,
        metadata=metadata,
        component=component,
    )
    metadata.update(metadata_update)

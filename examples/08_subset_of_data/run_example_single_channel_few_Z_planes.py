import os

from devtools import debug

from fractal_tasks_core.copy_ome_zarr import copy_ome_zarr
from fractal_tasks_core.create_ome_zarr import create_ome_zarr
from fractal_tasks_core.maximum_intensity_projection import (
    maximum_intensity_projection,
)
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


# Init
img_path = "../images/10.5281_zenodo.7057076/"
if not os.path.isdir(img_path):
    raise FileNotFoundError(
        f"{img_path} is missing,"
        " try running ./fetch_test_data_from_zenodo.sh"
    )
zarr_path = "tmp_out"
metadata = {}

# Create zarr structure
metadata_update = create_ome_zarr(
    input_paths=[img_path],
    output_path=zarr_path,
    metadata=metadata,
    image_extension="png",
    image_glob_patterns=["*A02*C03*", "*Z0[5-7]*"],
    allowed_channels=allowed_channels,
    num_levels=num_levels,
    coarsening_xy=coarsening_xy,
    metadata_table="mrf_mlf",
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
    metadata_update = maximum_intensity_projection(
        input_paths=[zarr_path],
        output_path=zarr_path,
        metadata=metadata,
        component=component,
    )
    metadata.update(metadata_update)

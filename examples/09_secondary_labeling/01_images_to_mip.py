import json
from typing import Any

from devtools import debug

from fractal_tasks_core.tasks.copy_ome_zarr import copy_ome_zarr
from fractal_tasks_core.tasks.create_ome_zarr import create_ome_zarr
from fractal_tasks_core.tasks.maximum_intensity_projection import (
    maximum_intensity_projection,
)
from fractal_tasks_core.tasks.yokogawa_to_ome_zarr import yokogawa_to_ome_zarr


allowed_channels = [
    {
        "wavelength_id": "A01_C01",
        "color": "00FFFF",
        "label": "Channel 1",
        "window": {"start": 110, "end": 2000},
    },
    {
        "wavelength_id": "A02_C02",
        "color": "FF00FF",
        "label": "Channel 2",
        "window": {"start": 110, "end": 500},
    },
    {
        "wavelength_id": "A03_C03",
        "color": "00FF00",
        "label": "Channel 3",
        "window": {"start": 110, "end": 1600},
    },
    {
        "wavelength_id": "A04_C04",
        "color": "FFFF00",
        "label": "Channel 4",
        "window": {"start": 110, "end": 1600},
    },
]


num_levels = 4
coarsening_xy = 2


# Init
img_path = "images/"
zarr_path = "output/"
zarr_path_mip = "output_mip/"
metadata: dict[str, Any] = {}

# Create zarr structure
metadata_update = create_ome_zarr(
    input_paths=[img_path],
    output_path=zarr_path,
    metadata=metadata,
    image_extension="tif",
    image_glob_patterns=["*A01*C01*"],
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
debug(metadata)

# Replicate
metadata_update = copy_ome_zarr(
    input_paths=[zarr_path],
    output_path=zarr_path_mip,
    metadata=metadata,
    project_to_2D=True,
    suffix="mip",
    input_ROI_tables=["well_ROI_table", "FOV_ROI_table"],
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
debug(metadata)
with open("01_final_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4, sort_keys=True)

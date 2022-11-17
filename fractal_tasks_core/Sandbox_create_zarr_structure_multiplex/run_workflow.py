import os
from pathlib import Path

from devtools import debug

from fractal_tasks_core.create_zarr_structure_multiplex import create_zarr_structure_multiplex
from fractal_tasks_core.yokogawa_to_zarr import yokogawa_to_zarr


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


# Init
#input_paths = [Path("./images/10.5281_zenodo.7059515/*.png")]
input_paths = list(folder / "*.png" for folder in Path("images/tiny_multiplexing/").glob("cycle*"))

zarr_path = Path("tmp_out/*.zarr")
metadata = {}

# Create zarr structure
metadata_update = create_zarr_structure_multiplex(
    input_paths=input_paths,
    output_path=zarr_path,
    channel_parameters=channel_parameters,
    num_levels=num_levels,
    coarsening_xy=coarsening_xy,
    metadata_table="mrf_mlf",
)
metadata.update(metadata_update)
debug(metadata)
assert metadata["image"]

# Yokogawa to zarr
for component in metadata["image"]:
    yokogawa_to_zarr(
        input_paths=[zarr_path],
        output_path=zarr_path,
        metadata=metadata,
        component=component,
    )
debug(metadata)

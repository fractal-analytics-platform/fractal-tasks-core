from pathlib import Path

from devtools import debug
from lib_validate_schema import validate_schema

from fractal_tasks_core.create_zarr_structure_multiplex import (
    create_zarr_structure_multiplex,
)
from fractal_tasks_core.yokogawa_to_zarr import yokogawa_to_zarr


channel_parameters_1 = {
    "A01_C01": {
        "label": "DAPI",
        "colormap": "00FFFF",
        "start": 0,
        "end": 700,
    },
}
channel_parameters_2 = {
    "A01_C01": {
        "label": "DAPI",
        "colormap": "00FFFF",
        "start": 0,
        "end": 700,
    },
    "A02_C03": {
        "label": "Na/K ATPase",
        "colormap": "FFFF00",
        "start": 0,
        "end": 1500,
    },
}
channel_parameters_3 = {
    "A01_C01": {
        "label": "DAPI",
        "colormap": "00FFFF",
        "start": 0,
        "end": 700,
    },
    "A01_C02": {
        "label": "HSP60",
        "colormap": "FF00FF",
        "start": 0,
        "end": 180,
    },
    "A02_C03": {
        "label": "Tubulin",
        "colormap": "FFFF00",
        "start": 0,
        "end": 1500,
    },
}
channel_parameters = {
    0: channel_parameters_1,
    1: channel_parameters_2,
    2: channel_parameters_3,
}

num_levels = 2
coarsening_xy = 2


# Init
input_paths = list(
    folder / "*.png"
    for folder in Path("images/tiny_multiplexing/").glob("cycle*")
)
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


# OME-NGFF JSON validation

image_zarr = Path(zarr_path.parent / metadata["image"][0])
well_zarr = image_zarr.parent
plate_zarr = image_zarr.parents[2]
validate_schema(path=str(image_zarr), type="image")
validate_schema(path=str(well_zarr), type="well")
validate_schema(path=str(plate_zarr), type="plate")


# Yokogawa to zarr
for component in metadata["image"]:
    yokogawa_to_zarr(
        input_paths=[zarr_path],
        output_path=zarr_path,
        metadata=metadata,
        component=component,
    )
debug(metadata)

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

from fractal_tasks_core.cellpose_segmentation import cellpose_segmentation
from fractal_tasks_core.create_ome_zarr import create_ome_zarr
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
img_path = "../images/10.5281_zenodo.7059515/"
if not os.path.isdir(Path(img_path).parent):
    raise FileNotFoundError(
        f"{Path(img_path).parent} is missing,"
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
debug(metadata)

# Per-FOV labeling
for component in metadata["image"]:
    cellpose_segmentation(
        input_paths=[zarr_path],
        output_path=zarr_path,
        metadata=metadata,
        component=component,
        wavelength_id="A01_C01",
        level=3,
        relabeling=True,
        diameter_level0=80.0,
    )
debug(metadata)

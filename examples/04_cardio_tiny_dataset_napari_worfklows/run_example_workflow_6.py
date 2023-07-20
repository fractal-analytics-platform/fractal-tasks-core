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

from fractal_tasks_core.lib_channels import OmeroChannel
from fractal_tasks_core.lib_channels import Window
from fractal_tasks_core.tasks.create_ome_zarr import create_ome_zarr
from fractal_tasks_core.tasks.napari_workflows_wrapper import (
    napari_workflows_wrapper,
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
    metadata=metadata,
    allowed_channels=allowed_channels,
    image_extension="png",
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

# napari-workflows
workflow_file = "wf_6.yaml"
input_specs = {
    "slice_img": {"type": "image", "channel": {"wavelength_id": "A01_C01"}},
    "slice_img_c2": {"type": "image", "channel": {"wavelength_id": "A01_C01"}},
}
output_specs = {
    "Result of Expand labels (scikit-image, nsbatwm)": {
        "type": "label",
        "label_name": "label_DAPI",
    },
    "regionprops_DAPI": {
        "type": "dataframe",
        "table_name": "regionprops_DAPI",
    },
}
for component in metadata["image"]:
    napari_workflows_wrapper(
        input_paths=[zarr_path],
        output_path=zarr_path,
        metadata=metadata,
        component=component,
        input_specs=input_specs,
        output_specs=output_specs,
        workflow_file=workflow_file,
        input_ROI_table="FOV_ROI_table",
    )
debug(metadata)

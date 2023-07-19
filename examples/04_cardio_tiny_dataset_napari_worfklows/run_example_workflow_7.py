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
from tmp_channels import allowed_channels

from fractal_tasks_core.tasks.cellpose_segmentation import (
    cellpose_segmentation,
)
from fractal_tasks_core.tasks.create_ome_zarr import create_ome_zarr
from fractal_tasks_core.tasks.napari_workflows_wrapper import (
    napari_workflows_wrapper,
)
from fractal_tasks_core.tasks.yokogawa_to_ome_zarr import yokogawa_to_ome_zarr


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

# Per-FOV labeling
for component in metadata["image"]:
    cellpose_segmentation(
        input_paths=[zarr_path],
        output_path=zarr_path,
        metadata=metadata,
        component=component,
        channel=dict(wavelength_id="A01_C01"),
        level=4,
        relabeling=True,
        diameter_level0=80.0,
    )
debug(metadata)

# napari-workflows
workflow_file = "wf_7.yaml"
input_specs = {
    "dapi_img": {"type": "image", "channel": {"wavelength_id": "A01_C01"}},
    "lamin_img": {"type": "image", "channel": {"wavelength_id": "A01_C01"}},
    "dapi_label_img": {"type": "label", "label_name": "label_DAPI"},
    "lamin_label_img": {"type": "label", "label_name": "label_DAPI"},
}
output_specs = {
    "regionprops_DAPI": {
        "type": "dataframe",
        "table_name": "regionprops_DAPI",
    },
    "regionprops_Lamin": {
        "type": "dataframe",
        "table_name": "regionprops_Lamin",
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

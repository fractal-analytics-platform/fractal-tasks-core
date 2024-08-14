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

from devtools import debug

from fractal_tasks_core.channels import OmeroChannel
from fractal_tasks_core.channels import Window
from fractal_tasks_core.tasks.cellpose_segmentation import (
    cellpose_segmentation,
)
from fractal_tasks_core.tasks.cellpose_transforms import (
    CellposeCustomNormalizer,
)
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
        window=Window(start=110, end=700),
    )
]


num_levels = 6
coarsening_xy = 2


# Init
img_path = "../images/10.5281_zenodo.7059515/"
if not os.path.isdir(img_path):
    raise FileNotFoundError(
        f"{img_path} is missing,"
        " try running ./fetch_test_data_from_zenodo.sh"
    )
zarr_path = "tmp_out_normalize/"
metadata: dict = {}

# Create zarr structure
metadata_update = create_ome_zarr(
    input_paths=[img_path],
    output_path=zarr_path,
    metadata=metadata,
    image_extension="png",
    allowed_channels=allowed_channels,
    num_levels=num_levels,
    coarsening_xy=coarsening_xy,
    overwrite=True,
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


# Copy zarr structure
metadata_update = copy_ome_zarr(
    input_paths=[zarr_path],
    output_path=zarr_path,
    metadata=metadata,
    suffix="mip",
    overwrite=True,
)
metadata.update(metadata_update)

# Make MIP
for component in metadata["image"]:
    metadata_update = projection(
        input_paths=[zarr_path],
        output_path=zarr_path,
        metadata=metadata,
        component=component,
    )
    metadata.update(metadata_update)


normalize = CellposeCustomNormalizer(default_normalize=True)
normalize = CellposeCustomNormalizer(
    default_normalize=False, lower_percentile=1.0, upper_percentile=99.0
)
# normalize=CellposeCustomNormalizer(
#     default_normalize=True,
#     lower_percentile=1.0,
#     upper_percentile=99.0
# ) # ValueError
# normalize=CellposeCustomNormalizer(
#     default_normalize=False,
#     lower_percentile=1.0,
#     upper_percentile=99.0,
#     lower_bound=100,
#     upper_bound=500
# ) # ValueError
normalize = CellposeCustomNormalizer(
    default_normalize=False, lower_bound=100, upper_bound=500
)


# Per-FOV labeling
for component in metadata["image"]:
    cellpose_segmentation(
        input_paths=[zarr_path],
        output_path=zarr_path,
        metadata=metadata,
        component=component,
        channel=dict(wavelength_id="A01_C01"),
        level=1,
        relabeling=True,
        diameter_level0=40.0,
        model_type="cyto2",
        normalize=normalize,
    )
    metadata.update(metadata_update)
debug(metadata)

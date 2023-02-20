import json
import os
import shutil
from pathlib import Path

from devtools import debug

from fractal_tasks_core.cellpose_secondary_segmentation import (
    cellpose_secondary_segmentation,
)  # noqa
from fractal_tasks_core.cellpose_segmentation import cellpose_segmentation


if os.path.exists("tmp"):
    shutil.rmtree("tmp")
os.mkdir("tmp")
shutil.copytree(
    "output/RS220304172545_mip.zarr", "tmp/RS220304172545_mip.zarr"
)

# Init
zarr_path_mip = Path("tmp/*zarr")
with open("01_final_metadata.json", "r") as f:
    metadata = json.load(f)
debug(metadata)


# Cellpose for organoids
for component in metadata["image"]:
    cellpose_segmentation(
        input_paths=[zarr_path_mip],
        output_path=zarr_path_mip,
        metadata=metadata,
        component=component,
        wavelength_id="A01_C01",
        level=2,
        relabeling=True,
        diameter_level0=400.0,
        ROI_table_name="well_ROI_table",
        cellprob_threshold=-3.0,
        flow_threshold=0.4,
        pretrained_model="model/Hummingbird.331986",
        output_label_name="organoids_ROI_table",
        bounding_box_ROI_table_name="organoids_bbox_table",
    )

print("\n--------------------\n")

for component in metadata["image"]:
    cellpose_secondary_segmentation(
        input_paths=[zarr_path_mip],
        output_path=zarr_path_mip,
        metadata=metadata,
        component=component,
        wavelength_id="A01_C01",
        level=2,
        relabeling=True,
        diameter_level0=20.0,
        ROI_table_name="organoids_bbox_table",
        primary_label_ROI_table_name="organoids_ROI_table",
        output_label_name="nuclei_ROI_table",
        model_type="nuclei",
        flow_threshold=0.4,
    )

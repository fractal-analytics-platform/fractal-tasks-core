# %%
from pathlib import Path

from ngio import create_synthetic_ome_zarr

from fractal_tasks_core.measure_features import measure_features
from fractal_tasks_core.threshold_segmentation import (
    InputChannel,
    threshold_segmentation,
)

path = Path("./test.img.zarr").resolve().as_posix()
print(path)

create_synthetic_ome_zarr(
    path,
    shape=(2, 2, 1, 512, 512),
    channels_meta=["DAPI", "GFP"],
    overwrite=True,
)
threshold_segmentation(
    zarr_url=path,
    channels=InputChannel(mode="index", identifier="0"),
)

measure_features(
    zarr_url=path,
    label_image_name="0_segmented",
    roi_tables=["well_ROI_table"],
    table_backend="csv",
)

# %%

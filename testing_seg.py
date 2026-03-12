# %%
from pathlib import Path

from ngio import create_synthetic_ome_zarr

from fractal_tasks_core.threshold_segmentation import (
    InputChannel,
    threshold_segmentation,
)

path = Path("./test.img.zarr").resolve().as_posix()
print(path)

create_synthetic_ome_zarr(
    path,
    shape=(1, 1, 512, 512),
    overwrite=True,
)
threshold_segmentation(
    zarr_url=path,
    channels=InputChannel(mode="index", identifier="0"),
)
# %%

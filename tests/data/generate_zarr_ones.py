import json
import os
import shutil

import dask.array as da
import numpy as np
import pandas as pd
import zarr

from fractal_tasks_core.roi import prepare_FOV_ROI_table
from fractal_tasks_core.tables import write_table


num_C = 2
num_Z = 2
num_Y = 2
num_X = 1
num_levels = 5
cxy = 2


x = da.ones(
    (num_C, num_Z, num_Y * 2160, num_X * 2560), chunks=(1, 1, 2160, 2560)
).astype(np.uint16)


zarrurl = "plate_ones.zarr/"

if os.path.isdir(zarrurl):
    shutil.rmtree(zarrurl)

plate_group = zarr.open_group(zarrurl)
plate_group.attrs.put(
    {
        "plate": {
            "acquisitions": [{"id": 1, "name": "plate_ones"}],
            "columns": [{"name": "03"}],
            "rows": [{"name": "B"}],
            "version": "0.4",
            "wells": [{"columnIndex": 0, "path": "B/03", "rowIndex": 0}],
        }
    }
)

well_group = zarr.open(f"{zarrurl}B/03/")
well_group.attrs.put({"well": {"images": [{"path": "0"}], "version": "0.4"}})

component = "B/03/0/"

for ind_level in range(num_levels):
    scale = 2**ind_level
    y = da.coarsen(np.min, x, {2: scale, 3: scale}).rechunk(
        (1, 1, 2160, 2560), balance=True
    )
    y.to_zarr(
        zarrurl, component=f"{component}{ind_level}", dimension_separator="/"
    )

pxl_x = 0.1625
pxl_y = 0.1625
pxl_z = 1.0

axes = [
    {"name": "c", "type": "channel"},
    {"name": "z", "type": "space", "unit": "micrometer"},
    {"name": "y", "type": "space", "unit": "micrometer"},
    {"name": "x", "type": "space", "unit": "micrometer"},
]

cT = "coordinateTransformations"
zattrs = {
    "multiscales": [
        {
            "axes": axes,
            "datasets": [
                {
                    "path": str(level),
                    cT: [
                        {
                            "type": "scale",
                            "scale": [
                                1.0,
                                pxl_z,
                                pxl_y * cxy**level,
                                pxl_x * cxy**level,
                            ],
                        }
                    ],
                }
                for level in range(num_levels)
            ],
            "version": "0.4",
        }
    ],
    "omero": {
        "channels": [
            {
                "wavelength_id": "A01_C01",
                "label": "some-label-1",
                "window": {"min": "0", "max": "10", "start": "0", "end": "10"},
                "color": "00FFFF",
            },
            {
                "wavelength_id": "A01_C02",
                "label": "some-label-2",
                "window": {"min": "0", "max": "10", "start": "0", "end": "10"},
                "color": "00FFFF",
            },
        ]
    },
}
with open(f"{zarrurl}{component}.zattrs", "w") as jsonfile:
    json.dump(zattrs, jsonfile, indent=4)


pixel_size_z = 1.0
pixel_size_y = 0.1625
pixel_size_x = 0.1625

df = pd.DataFrame(np.zeros((2, 10)), dtype=int)
df.index = ["FOV1", "FOV2"]
df.columns = [
    "x_micrometer_original",
    "y_micrometer_original",
    "z_micrometer",
    "x_pixel",
    "y_pixel",
    "z_pixel",
    "pixel_size_x",
    "pixel_size_y",
    "pixel_size_z",
    "bit_depth",
]
df["x_micrometer"] = [0.0, 0.0]
df["y_micrometer"] = [0.0, 351.0]
df["z_micrometer"] = [0.0, 0.0]
df["x_pixel"] = [2560] * 2
df["y_pixel"] = [2160] * 2
df["z_pixel"] = [num_Z] * 2
df["pixel_size_x"] = [pxl_x] * 2
df["pixel_size_y"] = [pxl_y] * 2
df["pixel_size_z"] = [pxl_z] * 2
df["bit_depth"] = [16.0, 16.0]


FOV_ROI_table = prepare_FOV_ROI_table(df)
print(FOV_ROI_table.to_df())

image_group = zarr.group(f"{zarrurl}{component}")
write_table(
    image_group,
    "FOV_ROI_table",
    FOV_ROI_table,
    overwrite=True,
    table_attrs=dict(fractal_table_version="1", type="roi_table"),
)

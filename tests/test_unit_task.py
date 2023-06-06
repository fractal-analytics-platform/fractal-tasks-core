import json
from pathlib import Path

from devtools import debug

import fractal_tasks_core
from fractal_tasks_core.tasks.create_ome_zarr import create_ome_zarr


# Load manifest
module_dir = Path(fractal_tasks_core.__file__).parent
with (module_dir / "__FRACTAL_MANIFEST__.json").open("r") as fin:
    __FRACTAL_MANIFEST__ = json.load(fin)

# Select a task
create_ome_zarr_manifest = next(
    item
    for item in __FRACTAL_MANIFEST__["task_list"]
    if item["name"] == "Create OME-Zarr structure"
)


def test_create_ome_zarr(tmp_path, testdata_path):
    input_paths = [str(testdata_path / "png/")]
    output_path = str(tmp_path)
    args = {}
    args["allowed_channels"] = [{"wavelength_id": "A01_C01"}]
    args["image_extension"] = "png"

    debug(input_paths)
    debug(output_path)
    debug(args)

    dummy = create_ome_zarr(
        input_paths=input_paths, output_path=output_path, metadata={}, **args
    )
    debug(dummy)

    zattrs = Path(output_path) / "myplate.zarr/.zattrs"
    with open(zattrs) as f:
        data = json.load(f)
        debug(data)
    assert len(data["plate"]["wells"]) == 1

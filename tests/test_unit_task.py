import json
from pathlib import Path

from devtools import debug

import fractal_tasks_core
from fractal_tasks_core.create_ome_zarr import create_ome_zarr


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
    input_paths = [testdata_path / "png/*.png"]
    output_path = tmp_path / "*.zarr"
    default_args = create_ome_zarr_manifest["default_args"]
    default_args["channel_parameters"] = {"A01_C01": {}}

    for key in ["executor", "parallelization_level"]:
        if key in default_args.keys():
            default_args.pop(key)

    from glob import glob

    debug(glob(input_paths[0].as_posix()))

    debug(input_paths)
    debug(output_path)
    debug(default_args)

    dummy = create_ome_zarr(
        input_paths=input_paths, output_path=output_path, **default_args
    )
    debug(dummy)

    debug(list(output_path.glob("*")))
    zattrs = output_path.parent / "myplate.zarr/.zattrs"
    with open(zattrs) as f:
        data = json.load(f)
        debug(data)
    assert len(data["plate"]["wells"]) == 1

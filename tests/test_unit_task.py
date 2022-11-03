import json

from devtools import debug

from fractal_tasks_core import __FRACTAL_MANIFEST__


create_zarr_structure_manifest = next(
    item
    for item in __FRACTAL_MANIFEST__
    if item["name"] == "Create OME-ZARR structure"
)


def test_create_zarr_structure(tmp_path, testdata_path):
    from fractal_tasks_core.create_zarr_structure import create_zarr_structure
    input_paths = [testdata_path / "png/*.png"]
    output_path = tmp_path / "*.zarr"
    default_args = create_zarr_structure_manifest["default_args"]
    default_args["channel_parameters"] = {"A01_C01": {}}

    for key in ["executor", "parallelization_level"]:
        if key in default_args.keys():
            default_args.pop(key)

    from glob import glob

    debug(glob(input_paths[0].as_posix()))

    debug(input_paths)
    debug(output_path)
    debug(default_args)

    dummy = create_zarr_structure(
        input_paths=input_paths, output_path=output_path, **default_args
    )
    debug(dummy)

    debug(list(output_path.glob("*")))
    zattrs = output_path.parent / "myplate.zarr/.zattrs"
    with open(zattrs) as f:
        data = json.load(f)
        debug(data)
    assert len(data["plate"]["wells"]) == 1

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
    args["allowed_channels"] = [
        {"wavelength_id": "A01_C01", "window": dict(start=0, end=1000)}
    ]
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


def test_run_fractal_tasks(tmp_path, testdata_path, monkeypatch):
    """
    Run a task funtion through run_fractal_task, after mocking the argparse
    interface
    """

    import fractal_tasks_core.tasks._utils

    # Write arguments to a file
    args = {}
    args["input_paths"] = [str(testdata_path / "png/")]
    args["output_path"] = str(tmp_path)
    args["allowed_channels"] = [
        {"wavelength_id": "A01_C01", "window": dict(start=0, end=1000)}
    ]
    args["image_extension"] = "png"
    args["metadata"] = {}
    debug(args)
    args_path = tmp_path / "args.json"
    with args_path.open("w") as f:
        json.dump(args, f, indent=2)

    # Mock argparse.ArgumentParser
    class MockArgumentParser:
        def add_argument(self, *args, **kwargs):
            pass

        def parse_args(self, *args, **kwargs):
            class Args(object):
                def __init__(self):
                    debug("INIT")
                    self.metadata_out = str(tmp_path / "metadiff.json")
                    self.json = str(args_path)

            return Args()

    monkeypatch.setattr(
        "fractal_tasks_core.tasks._utils.ArgumentParser",
        MockArgumentParser,
    )

    # Run the task
    out = fractal_tasks_core.tasks._utils.run_fractal_task(
        task_function=create_ome_zarr
    )

    # Check that the task wrote some output to args.metadata_out
    with (tmp_path / "metadiff.json").open("r") as f:
        out = json.load(f)
    debug(out)
    assert out

    # Check that the output zarr exists and includes a well
    zattrs = Path(args["output_path"]) / "myplate.zarr/.zattrs"
    with open(zattrs) as f:
        data = json.load(f)
        debug(data)
    assert len(data["plate"]["wells"]) == 1

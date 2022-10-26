import json
import subprocess
from json import JSONEncoder
from pathlib import Path

from devtools import debug

import fractal_tasks_core


channel_parameters = {
    "A01_C01": {
        "label": "DAPI",
        "colormap": "00FFFF",
        "start": 0,
        "end": 700,
    },
    "A01_C02": {
        "label": "nanog",
        "colormap": "FF00FF",
        "start": 0,
        "end": 180,
    },
    "A02_C03": {
        "label": "Lamin B1",
        "colormap": "FFFF00",
        "start": 0,
        "end": 1500,
    },
}

num_levels = 6
coarsening_xy = 2


class TaskParameterEncoder(JSONEncoder):
    def default(self, value):
        if isinstance(value, Path):
            return value.as_posix()
        return JSONEncoder.default(self, value)


def test_workflow_yokogawa_to_zarr(
    tmp_path: Path, dataset_10_5281_zenodo_7059515: Path
):

    # Init
    img_path = dataset_10_5281_zenodo_7059515 / "*.png"
    zarr_path = tmp_path / "tmp_out/*.zarr"
    metadata = {}

    # Create zarr structure
    args_create_zarr = dict(
        input_paths=[img_path],
        output_path=zarr_path,
        channel_parameters=channel_parameters,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        metadata_table="mrf_mlf",
    )

    with open(f"{str(tmp_path)}/args_create_zarr.json", "w") as js:
        json.dump(args_create_zarr, js, cls=TaskParameterEncoder)

    cmd = f"python {Path(fractal_tasks_core.__file__).parent}/create_zarr_structure.py \
            -j {str(tmp_path)}/args_create_zarr.json \
            --metadata-out {str(tmp_path)}/metadata_create_zarr.json"

    complete_create_zarr = subprocess.run(cmd, shell=True, check=False)
    debug(complete_create_zarr.stderr)
    debug(complete_create_zarr.stdout)

    with open(f"{str(tmp_path)}/metadata_create_zarr.json", "r") as js:
        diff_metadata = json.load(js)

    metadata.update(diff_metadata)
    debug(metadata)

    # Yokogawa to zarr
    for component in metadata["well"]:
        args_yokogawa = dict(
            input_paths=[zarr_path],
            output_path=zarr_path,
            metadata=metadata,
            component=component,
        )

        with open(f"{str(tmp_path)}/args_yokogawa.json", "w") as js:
            json.dump(args_yokogawa, js, cls=TaskParameterEncoder)

        cmd = f"python {Path(fractal_tasks_core.__file__).parent}/yokogawa_to_zarr.py \
                -j {str(tmp_path)}/args_yokogawa.json \
                --metadata-out {str(tmp_path)}/metadata_yokogawa.json"

        complete_yokogawa = subprocess.run(cmd, shell=True, check=False)
        debug(complete_yokogawa.stderr)
        debug(complete_yokogawa.stdout)

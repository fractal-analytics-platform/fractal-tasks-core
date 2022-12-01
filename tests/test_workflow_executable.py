import json
import subprocess
from json import JSONEncoder
from pathlib import Path

from devtools import debug

import fractal_tasks_core


allowed_channels = [
    {
        "label": "DAPI",
        "wavelength_id": "A01_C01",
        "colormap": "00FFFF",
        "start": 0,
        "end": 700,
    },
    {
        "wavelength_id": "A01_C02",
        "label": "nanog",
        "colormap": "FF00FF",
        "start": 0,
        "end": 180,
    },
    {
        "wavelength_id": "A02_C03",
        "label": "Lamin B1",
        "colormap": "FFFF00",
        "start": 0,
        "end": 1500,
    },
]


num_levels = 6
coarsening_xy = 2


class TaskParameterEncoder(JSONEncoder):
    def default(self, value):
        if isinstance(value, Path):
            return value.as_posix()
        return JSONEncoder.default(self, value)


def run_command(cmd: str):
    debug(cmd)
    completed_process = subprocess.run(cmd, shell=True, check=False)
    debug(completed_process.stdout)
    debug(completed_process.stderr)
    assert completed_process.returncode == 0
    assert completed_process.stderr is None


def test_workflow_yokogawa_to_ome_zarr(tmp_path: Path, zenodo_images: Path):

    # Init
    img_path = zenodo_images / "*.png"
    zarr_path = tmp_path / "tmp_out/*.zarr"
    metadata = {}
    tasks_path = str(Path(fractal_tasks_core.__file__).parent)

    # Create zarr structure
    args_create_zarr = dict(
        input_paths=[img_path],
        output_path=zarr_path,
        allowed_channels=allowed_channels,
        metadata={},
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        metadata_table="mrf_mlf",
    )

    # Run task as executable
    input_json_path = f"{str(tmp_path)}/args_create_zarr.json"
    output_json_path = f"{str(tmp_path)}/metadata_create_zarr.json"
    with open(input_json_path, "w") as js:
        json.dump(args_create_zarr, js, cls=TaskParameterEncoder)
    cmd = (
        f"python {tasks_path}/create_ome_zarr.py "
        f"-j {input_json_path} "
        f"--metadata-out {output_json_path}"
    )
    run_command(cmd)
    with open(output_json_path, "r") as js:
        diff_metadata = json.load(js)

    # Update metadata
    metadata.update(diff_metadata)
    debug(metadata)

    # Yokogawa to zarr
    for component in metadata["image"]:
        args_yokogawa = dict(
            input_paths=[zarr_path],
            output_path=zarr_path,
            metadata=metadata,
            component=component,
        )

        # Run task as executable
        input_json_path = f"{str(tmp_path)}/args_yokogawa.json"
        output_json_path = f"{str(tmp_path)}/metadata_yokogawa.json"
        with open(input_json_path, "w") as js:
            json.dump(args_yokogawa, js, cls=TaskParameterEncoder)
        cmd = (
            f"python {tasks_path}/yokogawa_to_ome_zarr.py "
            f"-j {input_json_path} "
            f"--metadata-out {output_json_path}"
        )
        run_command(cmd)

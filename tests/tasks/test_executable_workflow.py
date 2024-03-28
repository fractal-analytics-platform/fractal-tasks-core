import json
import subprocess
from json import JSONEncoder
from pathlib import Path

from devtools import debug

import fractal_tasks_core.tasks


allowed_channels = [
    {
        "label": "DAPI",
        "wavelength_id": "A01_C01",
        "color": "00FFFF",
        "window": {"start": 0, "end": 700},
    },
    {
        "wavelength_id": "A01_C02",
        "label": "nanog",
        "color": "FF00FF",
        "window": {"start": 0, "end": 180},
    },
    {
        "wavelength_id": "A02_C03",
        "label": "Lamin B1",
        "color": "FFFF00",
        "window": {"start": 0, "end": 1500},
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


def test_workflow_yokogawa_to_ome_zarr(tmp_path: Path, zenodo_images: str):

    # Init
    img_path = zenodo_images
    zarr_dir = str(tmp_path / "tmp_out/")
    tasks_path = str(Path(fractal_tasks_core.tasks.__file__).parent)

    # Create zarr structure
    args_create_zarr = dict(
        zarr_urls=[],
        zarr_dir=zarr_dir,
        image_dirs=[img_path],
        allowed_channels=allowed_channels,
        image_extension="png",
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        metadata_table_file=None,
    )

    # Run task as executable
    input_json_path = f"{str(tmp_path)}/args_create_zarr.json"
    output_json_path = f"{str(tmp_path)}/out_create_zarr.json"
    with open(input_json_path, "w") as js:
        json.dump(args_create_zarr, js, cls=TaskParameterEncoder)
    cmd = (
        f"python {tasks_path}/create_cellvoyager_ome_zarr_init.py "
        f"-j {input_json_path} "
        f"--metadata-out {output_json_path}"
    )
    run_command(cmd)
    with open(output_json_path, "r") as js:
        parallelization_list = json.load(js)
    debug(parallelization_list)

    # Yokogawa to zarr
    for image in parallelization_list:
        args_yokogawa = dict(
            zarr_url=image["zarr_url"],
            init_args=image["init_args"],
        )

        # Run task as executable
        input_json_path = f"{str(tmp_path)}/args_yokogawa.json"
        output_json_path = f"{str(tmp_path)}/out_yokogawa.json"
        with open(input_json_path, "w") as js:
            json.dump(args_yokogawa, js, cls=TaskParameterEncoder)
        cmd = (
            f"python {tasks_path}/create_cellvoyager_ome_zarr_compute.py "
            f"-j {input_json_path} "
            f"--metadata-out {output_json_path}"
        )
        run_command(cmd)

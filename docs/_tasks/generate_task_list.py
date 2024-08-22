import json
import logging
from pathlib import Path

import mkdocs_gen_files
import requests

logger = logging.getLogger(f"mkdocs.plugins.{__name__}")
prefix = f"[{Path(__file__).name}]"
logger.info(f"{prefix} START")


pkgs = dict()
pkgs["fractal-tasks-core"] = dict(
    homepage_url="https://fractal-analytics-platform.github.io/fractal-tasks-core",  # noqa
    manifest_url="https://raw.githubusercontent.com/fractal-analytics-platform/fractal-tasks-core/main/fractal_tasks_core/__FRACTAL_MANIFEST__.json",  # noqa
    description=(
        "The Fractal tasks core package is the reference implementation for "
        "Fractal tasks. It contains tasks to convert Cellvoyager CV7000 and "
        "CV8000 images to OME-Zarr, to make 3D projections, apply flatfield "
        "illumination corrections, segment objects using Cellpose, run "
        "napari workflows, calculate & apply registration and to import "
        "OME-Zarrs into a Fractal workflow."
    ),
)

pkgs["scMultiplex"] = dict(
    homepage_url="https://github.com/fmi-basel/gliberal-scMultipleX",
    manifest_url="https://raw.githubusercontent.com/fmi-basel/gliberal-scMultipleX/main/src/scmultiplex/__FRACTAL_MANIFEST__.json",  # noqa
    description=(
        "The scMultipleX package contains tasks to perform object-based "
        "registration, multiplexed measurements, mesh generations and more."
    ),
)
pkgs["fractal-faim-ipa"] = dict(
    homepage_url="https://github.com/fractal-analytics-platform/fractal-faim-ipa",  # noqa
    manifest_url="https://raw.githubusercontent.com/jluethi/fractal-faim-ipa/main/src/fractal_faim_ipa/__FRACTAL_MANIFEST__.json",  # noqa
    description=(
        "Provides Fractal tasks for the conversion of Molecular Devices "
        "ImageXpress microscope to OME-Zarr. This package is based on the "
        "faim-ipa library developed by FAIM at FMI."
    ),
)
pkgs["fractal-helper-tasks"] = dict(
    homepage_url="https://github.com/fractal-analytics-platform/fractal-helper-tasks",  # noqa
    manifest_url="https://raw.githubusercontent.com/fractal-analytics-platform/fractal-helper-tasks/main/src/fractal_helper_tasks/__FRACTAL_MANIFEST__.json",  # noqa
    description=("Collection of Fractal helper tasks."),
)
pkgs["APx_fractal_task_collection"] = dict(
    homepage_url="https://github.com/Apricot-Therapeutics/APx_fractal_task_collection",  # noqa
    manifest_url="https://raw.githubusercontent.com/Apricot-Therapeutics/APx_fractal_task_collection/main/src/apx_fractal_task_collection/__FRACTAL_MANIFEST__.json",  # noqa
    description=(
        "The APx Fractal Task Collection is mainainted by Apricot "
        "Therapeutics AG, Switzerland. This is a collection of tasks intended "
        "to be used in combination with the Fractal Analytics Platform "
        "maintained by the BioVisionCenter Zurich (co-founded by the "
        "Friedrich Miescher Institute and the University of Zurich). The "
        "tasks in this collection are focused on extending Fractal's "
        "capabilities of processing 2D image data, with a special focus on "
        "multiplexed 2D image data. Most tasks work with 3D image data, but "
        "they have not specifically been developed for this scenario."
    ),
)
pkgs["operetta-compose"] = dict(
    homepage_url="https://github.com/leukemia-kispi/operetta-compose",
    manifest_url="https://raw.githubusercontent.com/leukemia-kispi/operetta-compose/main/src/operetta_compose/__FRACTAL_MANIFEST__.json",  # noqa
    description=(
        "Fractal tasks for the Opera/Operetta "
        "microscope and drug response profiling."
    ),
)
pkgs["fractal-plantseg-tasks"] = dict(
    homepage_url="https://github.com/fractal-analytics-platform/fractal-plantseg-tasks",  # noqa
    manifest_url="https://raw.githubusercontent.com/fractal-analytics-platform/fractal-plantseg-tasks/main/src/plantseg_tasks/__FRACTAL_MANIFEST__.json",  # noqa
    description=(
        "Collection of Fractal task with the PlantSeg segmentation pipeline."
    ),
)
pkgs["fractal-ome-zarr-hcs-stitching"] = dict(
    homepage_url="https://github.com/m-albert/fractal-ome-zarr-hcs-stitching",  # noqa
    manifest_url="https://raw.githubusercontent.com/m-albert/fractal-ome-zarr-hcs-stitching/main/src/fractal_ome_zarr_hcs_stitching/__FRACTAL_MANIFEST__.json",  # noqa
    description=(
        "Fractal task(s) for registering and fusing OME-Zarr HCS using "
        "multiview-stitcher."
    ),
)
pkgs["abbott"] = dict(
    homepage_url="https://github.com/MaksHess/abbott",
    manifest_url="https://raw.githubusercontent.com/MaksHess/abbott/main/src/abbott/__FRACTAL_MANIFEST__.json",  # noqa
    description=(
        "Work in progress task package to provide 3D registration methods "
        "based on Elastix."
    ),
)

script_path = __file__
script_dir = Path(script_path).parent
markdown_file = script_dir / "_all.md"
logger.info(f"{prefix} Writing output to {markdown_file}")

with mkdocs_gen_files.open(markdown_file.as_posix(), "w") as md:
    for package_name, package in pkgs.items():
        homepage_url = package["homepage_url"]
        manifest_url = package["manifest_url"]
        description = package.get("description", None)
        r = requests.get(manifest_url)
        if not r.status_code == 200:
            error_msg = f"Something wrong with the request to {manifest_url}"
            logger.error(f"{prefix} {error_msg}")
            raise ValueError(error_msg)
        manifest = json.loads(r.content.decode("utf-8"))
        task_list = manifest["task_list"]
        md.write(f"## `{package_name}`\n")
        md.write(f"**Package:** `{package_name}`\n\n")
        md.write(f"**Home page:** {homepage_url}\n\n")
        if description is not None:
            md.write(f"**Description:** {description}\n\n")
        md.write("**Tasks:**\n\n")
        for task in task_list:
            name = task["name"]
            docs_link = task.get("docs_link")
            if package_name == "fractal-tasks-core":
                md.write(f"* [{name}]({docs_link})\n")
            else:
                md.write(f"* {name}\n")
        num_tasks = len(task_list)
        logger.info(
            f"{prefix} Processed {package_name}, found {num_tasks} tasks"
        )
        md.write("\n\n")

logger.info(f"{prefix} END")

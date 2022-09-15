import logging
import os
import shutil
import subprocess  # nosec
from logging import FileHandler
from logging import Formatter
from logging import getLogger
from pathlib import Path

from fractal_tasks_core.illumination_correction import illumination_correction


def get_git_revision_hash() -> str:
    # See https://stackoverflow.com/a/21901260/19085332
    return (
        subprocess.check_output(["git", "rev-parse", "HEAD"])  # nosec
        .decode("ascii")
        .strip()
    )


# Handle logging
if os.path.isfile("logs"):
    os.remove("logs")
logger = getLogger("test_illum_corr")
formatter = Formatter("%(asctime)s; %(levelname)s; %(message)s")
handler = FileHandler("logs", mode="a")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logging.getLogger().setLevel(logging.INFO)

logging.info(
    f"git hash: {get_git_revision_hash()} (but there could be local changes)"
)


# Copy a reference zarr file
source_dir = "Backup_data/one_well_9x8_fovs_19_z_planes_3_channels.zarr"
shutil.rmtree("data")
os.makedirs("data")
os.makedirs("data/plate.zarr")
destination_dir = "data/plate.zarr"
shutil.rmtree(destination_dir)
shutil.copytree(source_dir, destination_dir)
logging.info("Input zarr file copied over")

# Set dataset path
zarr_path = Path("data/*.zarr")

# Set minimalistic metadata
metadata = {
    "well": ["plate.zarr/B/03/0/"],
    "num_levels": 5,
    "coarsening_xy": 2,
    "channel_list": ["A01_C01", "A01_C02", "A02_C03"],
}

# Set parameters
cwd = Path(__file__).parent.resolve().as_posix()
dict_corr = {
    "root_path_corr": f"{cwd}/parameters",
    "A01_C01": "illum_corr_matrix.png",
    "A01_C02": "illum_corr_matrix.png",
    "A02_C03": "illum_corr_matrix.png",
}


# Illumination correction
for component in metadata["well"]:
    illumination_correction(
        input_paths=[zarr_path],
        output_path=zarr_path,
        metadata=metadata,
        component=component,
        overwrite=True,
        dict_corr=dict_corr,
        background=100,
        logger=logger,
    )

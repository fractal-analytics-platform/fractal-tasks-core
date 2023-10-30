import json
import logging
import os
import shutil
import time
from pathlib import Path

import anndata as ad
import pytest
import requests  # type: ignore
import wget
import zarr
from devtools import debug

from fractal_tasks_core.lib_regions_of_interest import reset_origin
from fractal_tasks_core.lib_write import write_table


@pytest.fixture(scope="session")
def testdata_path() -> Path:
    TEST_DIR = Path(__file__).parent
    return TEST_DIR / "data/"


@pytest.fixture(scope="session")
def zenodo_images(testdata_path):
    """
    Inspired by
    https://github.com/dvolgyes/zenodo_get/blob/master/zenodo_get/zget.py
    """
    t_start = time.perf_counter()

    # Download images and metadata files
    recordID = "7059515"
    url = "10_5281_zenodo_7059515"
    folder = str(testdata_path / f"10_5281_zenodo_{recordID}")
    if os.path.isdir(folder):
        print(f"{folder} already exists, skip download")
    else:
        os.makedirs(folder)
        url = f"https://zenodo.org/api/records/{recordID}"
        r = requests.get(url)
        js = json.loads(r.text)
        files = js["files"]
        for f in files:
            file_url = f["links"]["self"]
            file_name = file_url.split("/")[-2]
            wget.download(file_url, out=f"{folder}/{file_name}", bar=False)

    # Add an image with invalid name, that should be skipped during parsing
    with open(f"{folder}/invalid_path.png", "w") as f:
        f.write("This file has an invalid filename, which cannot be parsed.")

    t_end = time.perf_counter()
    logging.warning(
        f"\n    Time spent in zenodo_images: {t_end-t_start:.2f} s"
    )

    return folder


@pytest.fixture(scope="session")
def zenodo_images_multiplex(testdata_path, zenodo_images):
    folder = str(testdata_path / "fake_multiplex")
    cycle_folder_1 = str(Path(folder) / "cycle1")
    cycle_folder_2 = str(Path(folder) / "cycle2")
    cycle_folders = [cycle_folder_1, cycle_folder_2]
    if os.path.isdir(folder):
        print(f"{folder} already exists, skip zenodo_images_multiplex")
    else:
        os.makedirs(folder)
        for cycle_folder in cycle_folders:
            shutil.copytree(zenodo_images, cycle_folder)
    return cycle_folders


@pytest.fixture(scope="session")
def zenodo_zarr(testdata_path, tmpdir_factory):
    t_start = time.perf_counter()

    doi = "10.5281/zenodo.8091756"
    rootfolder = testdata_path / (doi.replace(".", "_").replace("/", "_"))
    platenames = ["plate.zarr", "plate_mip.zarr"]
    folders = [rootfolder / plate for plate in platenames]
    zarrnames = [
        "20200812-CardiomyocyteDifferentiation14-Cycle1.zarr",
        "20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr",
    ]

    # Download dataset
    if rootfolder.exists():
        print(f"{str(rootfolder)} already exists, skip download part")
    else:
        rootfolder.mkdir()
        tmp_path = tmpdir_factory.mktemp("zenodo_zarr")
        for zarrname in zarrnames:
            zipname = f"{zarrname}.zip"
            url = f"https://zenodo.org/record/8091756/files/{zipname}"
            debug(url)
            wget.download(url, out=str(tmp_path / zipname), bar=None)
            time.sleep(0.5)
            shutil.unpack_archive(
                str(tmp_path / zipname),
                extract_dir=rootfolder,
                format="zip",
            )

    # Based on the Zenodo OME-Zarrs, create the appropriate OME-Zarrs to be
    # used in tests
    for zarrname, folder in zip(zarrnames, folders):
        if os.path.isdir(str(folder)):
            shutil.rmtree(str(folder))
        shutil.copytree(str(rootfolder / zarrname), str(folder))

        # Update well/FOV ROI tables, by shifting their origin to 0
        # TODO: remove this fix, by uploading new zarrs to zenodo (ref
        # issue 526)
        image_group_path = folder / "B/03/0"
        group_image = zarr.open_group(str(image_group_path))
        for table_name in ["FOV_ROI_table", "well_ROI_table"]:
            table_path = str(image_group_path / "tables" / table_name)
            old_table = ad.read_zarr(table_path)
            new_table = reset_origin(old_table)
            write_table(
                group_image,
                table_name,
                new_table,
                overwrite=True,
                logger=logging.getLogger(),
            )

    folders = [str(f) for f in folders]

    t_end = time.perf_counter()
    logging.warning(f"\n    Time spent in zenodo_zarr: {t_end-t_start:.2f} s")
    return folders


@pytest.fixture(scope="function")
def zenodo_zarr_metadata(testdata_path):
    metadata_3D = {
        "plate": ["plate.zarr"],
        "well": ["plate.zarr/B/03"],
        "image": ["plate.zarr/B/03/0/"],
        "num_levels": 6,
        "coarsening_xy": 2,
        "original_paths": [str(testdata_path / "10_5281_zenodo_7059515/")],
        "image_extension": "png",
    }

    metadata_2D = {
        "plate": ["plate.zarr"],
        "well": ["plate_mip.zarr/B/03/"],
        "image": ["plate_mip.zarr/B/03/0/"],
        "num_levels": 6,
        "coarsening_xy": 2,
        "original_paths": [str(testdata_path / "10_5281_zenodo_7059515/")],
        "image_extension": "png",
        "replicate_zarr": {
            "suffix": "mip",
            "sources": {"plate_mip": "/this/should/not/be/used/"},
        },
    }

    return [metadata_3D, metadata_2D]

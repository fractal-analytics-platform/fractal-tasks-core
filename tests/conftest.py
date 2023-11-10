import json
import logging
import os
import shutil
import time
from pathlib import Path

import anndata as ad
import pooch
import pytest
import requests  # type: ignore
import wget
import zarr

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
def zenodo_zarr(testdata_path):
    """
    This takes care of two steps:

    1. Download, via pooch
    2. Store a copy in tests/data
    3. Modify the copy in tests/data, to add whatever is not in Zenodo
    """

    DOI = "10.5281/zenodo.8091756"
    DOI_slug = DOI.replace("/", "_").replace(".", "_")
    platenames = ["plate.zarr", "plate_mip.zarr"]
    rootfolder = testdata_path / DOI_slug
    folders = [rootfolder / plate for plate in platenames]

    for ind, (file_name, known_hash) in enumerate(
        [
            (
                "20200812-CardiomyocyteDifferentiation14-Cycle1.zarr",
                "38b7894530f28fd6f55edf5272aaea104c11f36e28825446d23aff280f3a4290",  # noqa
            ),
            (
                "20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr",
                "7efddc0bd20b186c28ca8373b1f7af2d1723d0663fe9438969dc79da02539175",  # noqa
            ),
        ]
    ):
        file_paths = pooch.retrieve(
            url=f"doi:{DOI}/{file_name}.zip",
            fname=f"{file_name}.zip",
            known_hash=known_hash,
            processor=pooch.Unzip(extract_dir=f"{DOI_slug}/{file_name}"),
        )
        zarr_full_path = file_paths[0].split(file_name)[0] + file_name
        print(zarr_full_path)
        folder = folders[ind]

        # Based on the Zenodo OME-Zarrs, create the appropriate OME-Zarrs to be
        # used in tests
        if os.path.isdir(str(folder)):
            shutil.rmtree(str(folder))
        shutil.copytree(zarr_full_path, folder)

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

import json
import logging
import os
import shutil
from pathlib import Path
from urllib.parse import unquote

import pytest
import requests  # type: ignore
import wget


@pytest.fixture(scope="session")
def testdata_path() -> Path:
    TEST_DIR = Path(__file__).parent
    return TEST_DIR / "data/"


@pytest.fixture(scope="session")
def zenodo_images(testdata_path):
    # Based on
    # https://github.com/dvolgyes/zenodo_get/blob/master/zenodo_get/zget.py

    url = "10.5281/zenodo.7059515"
    folder = str(testdata_path / (url.replace(".", "_").replace("/", "_")))
    if os.path.isdir(folder):
        print(f"{folder} already exists, skip")
        return folder
    os.makedirs(folder)
    url = "https://doi.org/" + url
    print(f"I will download {url} files to {folder}")

    r = requests.get(url)
    recordID = r.url.split("/")[-1]
    url = "https://zenodo.org/api/records/"
    r = requests.get(url + recordID)

    js = json.loads(r.text)
    files = js["files"]
    for f in files:
        fname = f["key"]
        link = f"https://zenodo.org/record/{recordID}/files/{fname}"
        print(link)
        link = unquote(link)
        wget.download(link, out=folder)
        print()

    # Add an image with invalid name, that should be skipped during parsing
    with open(f"{folder}/invalid_path.png", "w") as f:
        f.write("This file has an invalid filename, which cannot be parsed.")

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

    doi = "10.5281/zenodo.7674545"
    rootfolder = testdata_path / (doi.replace(".", "_").replace("/", "_"))
    platenames = ["plate.zarr", "plate_mip.zarr"]
    folders = [rootfolder / plate for plate in platenames]

    if rootfolder.exists():
        print(f"{str(rootfolder)} already exists, skip")
    else:
        rootfolder.mkdir()
        tmp_path = tmpdir_factory.mktemp("zenodo_zarr")
        zarrnames = [
            "20200812-CardiomyocyteDifferentiation14-Cycle1.zarr",
            "20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr",
        ]
        for zarrname, folder in zip(zarrnames, folders):
            zipname = f"{zarrname}.zip"
            url = f"https://zenodo.org/record/7674545/files/{zipname}"
            wget.download(url, out=str(tmp_path / zipname), bar=None)
            shutil.unpack_archive(
                str(tmp_path / zipname), extract_dir=rootfolder, format="zip"
            )
            shutil.move(str(rootfolder / zarrname), str(folder))

            # Fix a wrong piece of metadata
            zattrs_path = folder / "B/03/0/.zattrs"
            logging.warning(
                f"Update coordinateTransformations in {str(zattrs_path)}, "
                "see https://github.com/fractal-analytics-platform/"
                "fractal-tasks-core/issues/420."
            )
            with zattrs_path.open("r") as f:
                zattrs = json.load(f)
            for ind, ds in enumerate(zattrs["multiscales"][0]["datasets"]):
                new_ds = ds.copy()
                old_transf = ds["coordinateTransformations"][0]
                new_transf = old_transf.copy()
                assert old_transf["type"] == "scale"
                assert len(old_transf["scale"]) == 3
                new_transf["scale"] = [1.0, *old_transf["scale"]]
                new_ds["coordinateTransformations"][0] = new_transf
                assert len(new_transf["scale"]) == len(
                    zattrs["multiscales"][0]["axes"]
                )
                zattrs["multiscales"][0]["datasets"][ind] = new_ds
            with zattrs_path.open("w") as f:
                json.dump(zattrs, f, indent=2)

    folders = [str(f) for f in folders]

    return folders


@pytest.fixture(scope="session")
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

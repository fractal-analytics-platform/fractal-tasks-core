import os
import shutil
from pathlib import Path

import pooch
import pytest
from devtools import debug

from ..conftest import *  # noqa


@pytest.fixture(scope="session")
def zenodo_images(testdata_path: Path) -> str:
    """
    1. Download image/metadata files from Zenodo;
    2. Copy image/metadata files into a tests/data subfolder;
    3. Add a spurious file.
    """

    DOI = "10.5281/zenodo.8287221"
    DOI_slug = DOI.replace("/", "_").replace(".", "_")
    rootfolder = testdata_path / DOI_slug
    rootfolder.mkdir(exist_ok=True)

    registry = {
        "20200812-CardiomyocyteDifferentiation14-Cycle1_B03_T0001F001L01A01Z01C01.png": "md5:41c5d3612f166d30d694a6c9902a5839",  # noqa
        "20200812-CardiomyocyteDifferentiation14-Cycle1_B03_T0001F001L01A01Z02C01.png": "md5:3aa92682cf731989cf4d3e0015f59ce0",  # noqa
        "20200812-CardiomyocyteDifferentiation14-Cycle1_B03_T0001F002L01A01Z01C01.png": "md5:a3b0be2af486e08d1f009831d8656b80",  # noqa
        "20200812-CardiomyocyteDifferentiation14-Cycle1_B03_T0001F002L01A01Z02C01.png": "md5:f1e0d50a1654ffd079504a036ff4a9e3",  # noqa
        "MeasurementData.mlf": "md5:08898b37193727874b45c65a11754db9",
        "MeasurementDetail.mrf": "md5:5fce4ca3e5ebc5f5be0b4945598e1ffb",
    }
    base_url = f"doi:{DOI}"
    POOCH = pooch.create(
        pooch.os_cache("pooch") / DOI_slug,
        base_url,
        registry=registry,
        retry_if_failed=10,
        allow_updates=False,
    )

    # Download files one by one, and copy them into rootfolder
    debug(rootfolder)

    for file_name in registry.keys():
        debug(file_name)
        file_path = POOCH.fetch(file_name)
        shutil.copy(file_path, rootfolder / file_name)

    # Add an image with invalid name, that should be skipped during parsing
    with (rootfolder / "invalid_path.png").open("w") as f:
        f.write("This file has an invalid filename, which cannot be parsed.")

    return rootfolder.as_posix()


@pytest.fixture(scope="session")
def zenodo_images_multiplex(testdata_path: Path, zenodo_images: Path):
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
def zenodo_zarr(testdata_path: Path) -> list[str]:
    """
    This takes care of multiple steps:

    1. Download/unzip two Zarr containers (3D and MIP) from Zenodo, via pooch
    2. Copy the two Zarr containers into tests/data
    3. Modify the Zarrs in tests/data, to add whatever is not in Zenodo
    """

    # 1 Download Zarrs from Zenodo
    DOI = "10.5281/zenodo.8091756"
    DOI_slug = DOI.replace("/", "_").replace(".", "_")
    platenames = ["plate.zarr", "plate_mip.zarr"]
    rootfolder = testdata_path / DOI_slug
    folders = [rootfolder / plate for plate in platenames]

    registry = {
        "20200812-CardiomyocyteDifferentiation14-Cycle1.zarr.zip": None,
        "20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr.zip": None,
    }
    base_url = f"doi:{DOI}"
    POOCH = pooch.create(
        pooch.os_cache("pooch") / DOI_slug,
        base_url,
        registry=registry,
        retry_if_failed=10,
        allow_updates=False,
    )

    for ind, file_name in enumerate(
        [
            "20200812-CardiomyocyteDifferentiation14-Cycle1.zarr",
            "20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr",
        ]
    ):
        # 1) Download/unzip a single Zarr from Zenodo
        file_paths = POOCH.fetch(
            f"{file_name}.zip", processor=pooch.Unzip(extract_dir=file_name)
        )
        zarr_full_path = file_paths[0].split(file_name)[0] + file_name
        print(zarr_full_path)
        folder = folders[ind]

        # 2) Copy the downloaded Zarr into tests/data
        if os.path.isdir(str(folder)):
            shutil.rmtree(str(folder))
        shutil.copytree(Path(zarr_full_path) / file_name, folder)

    return [str(f) for f in folders]


@pytest.fixture(scope="function")
def zenodo_zarr_metadata(testdata_path: Path):
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

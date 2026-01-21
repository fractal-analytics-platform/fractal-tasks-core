import os
import shutil
from pathlib import Path

import pooch
import pytest

from ..conftest import *  # noqa

ZENODO_HEADERS = {
    "User-Agent": "pooch (https://github.com/fatiando/pooch) "
    "(https://github.com/fractal-analytics-platform/fractal-tasks-core)",
    "Accept": "*/*",
}

ZENODO_DOWNLOADER = pooch.HTTPDownloader(headers=ZENODO_HEADERS, timeout=120)


@pytest.fixture(scope="session")
def zenodo_images(testdata_path: Path) -> str:
    """
    1. Download image/metadata files from Zenodo;
    2. Copy image/metadata files into a tests/data subfolder;
    3. Add a spurious file.
    """
    doi = "10.5281/zenodo.8287221"
    doi_slug = doi.replace("/", "_").replace(".", "_")
    rootfolder = testdata_path / doi_slug
    rootfolder.mkdir(exist_ok=True)

    record_id = "8287221"
    base_url = f"https://zenodo.org/records/{record_id}/files/"

    registry = {
        "20200812-CardiomyocyteDifferentiation14-Cycle1_B03_T0001F001L01A01Z01C01.png": "md5:41c5d3612f166d30d694a6c9902a5839",  # noqa: E501
        "20200812-CardiomyocyteDifferentiation14-Cycle1_B03_T0001F001L01A01Z02C01.png": "md5:3aa92682cf731989cf4d3e0015f59ce0",  # noqa: E501
        "20200812-CardiomyocyteDifferentiation14-Cycle1_B03_T0001F002L01A01Z01C01.png": "md5:a3b0be2af486e08d1f009831d8656b80",  # noqa: E501
        "20200812-CardiomyocyteDifferentiation14-Cycle1_B03_T0001F002L01A01Z02C01.png": "md5:f1e0d50a1654ffd079504a036ff4a9e3",  # noqa: E501
        "MeasurementData.mlf": "md5:08898b37193727874b45c65a11754db9",
        "MeasurementDetail.mrf": "md5:5fce4ca3e5ebc5f5be0b4945598e1ffb",
    }

    pup = pooch.create(
        path=pooch.os_cache("pooch") / doi_slug,
        base_url=base_url,
        registry=registry,
        retry_if_failed=10,
        allow_updates=False,
    )

    for file_name in registry:
        file_path = pup.fetch(file_name, downloader=ZENODO_DOWNLOADER)
        shutil.copy(file_path, rootfolder / file_name)

    (rootfolder / "invalid_path.png").write_text(
        "This file has an invalid filename, which cannot be parsed."
    )

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
    1. Download/unzip two Zarr containers (3D and MIP) from Zenodo, via pooch
    2. Copy the two Zarr containers into tests/data/<DOI_slug>/
    {plate.zarr,plate_mip.zarr}
    """
    DOI = "10.5281/zenodo.13305156"
    DOI_slug = DOI.replace("/", "_").replace(".", "_")

    platenames = ["plate.zarr", "plate_mip.zarr"]
    rootfolder = testdata_path / DOI_slug
    folders = [rootfolder / plate for plate in platenames]

    record_id = "13305156"
    base_url = f"https://zenodo.org/records/{record_id}/files/"
    # If you ever see flaky 403s again, try:
    # base_url = f"https://zenodo.org/records/{record_id}/files/?download=1"
    # Better is to add ?download=1 per file via `urls=` (see below),
    # but keep minimal for now.

    # pin checksums to avoid extra requests
    registry = {
        "20200812-CardiomyocyteDifferentiation14-Cycle1.zarr.zip": "md5:efc21fe8d4ea3abab76226d8c166452c",  # noqa: E501
        "20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr.zip": "md5:51809479777cafbe9ac0f9fa5636aa95",  # noqa: E501
    }

    POOCH = pooch.create(
        path=pooch.os_cache("pooch") / DOI_slug,
        base_url=base_url,
        registry=registry,
        retry_if_failed=10,
        allow_updates=False,
    )

    zarr_names = [
        "20200812-CardiomyocyteDifferentiation14-Cycle1.zarr",
        "20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr",
    ]

    for ind, zarr_name in enumerate(zarr_names):
        zip_name = f"{zarr_name}.zip"

        # Download/unzip a single Zarr from Zenodo
        file_paths = POOCH.fetch(
            zip_name,
            downloader=ZENODO_DOWNLOADER,
            processor=pooch.Unzip(extract_dir=zarr_name),
        )

        # Pooch returns a list of extracted paths; derive the folder
        # containing the .zarr
        zarr_full_path = file_paths[0].split(zarr_name)[0] + zarr_name
        folder = folders[ind]

        # Copy the downloaded Zarr into tests/data (replace if existing)
        if os.path.isdir(str(folder)):
            shutil.rmtree(str(folder))
        folder.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(Path(zarr_full_path) / zarr_name, folder)

    return [str(f) for f in folders]


@pytest.fixture(scope="session")
def syn_1536_images(testdata_path: Path, zenodo_images: str) -> str:
    """
    0. Check if images are already present;
    1. Copy images of the zenodo example to create synthetic images of a 1536
    well plate in the `testdata_path`/data/1536_well folder;
    2. Change the filenames to match the 1536 well convention;
    """
    target_folder = testdata_path / "syn_1536_images"
    target_files = [
        (
            "20200812-CardiomyocyteDifferentiation14-Cycle1_B03.a1_T0001F001L"
            "01A01Z01C01.png"
        ),
        (
            "20200812-CardiomyocyteDifferentiation14-Cycle1_B03.a1_T0001F002L"
            "01A01Z01C01.png"
        ),
    ]
    files_present = [f in os.listdir(target_folder) for f in target_files]
    if not all(files_present):
        shutil.copy(
            Path(zenodo_images)
            / (
                "20200812-CardiomyocyteDifferentiation14-Cycle1_B03_"
                "T0001F001L01A01Z01C01.png"
            ),
            target_folder / target_files[0],
        )
        shutil.copy(
            Path(zenodo_images)
            / (
                "20200812-CardiomyocyteDifferentiation14-Cycle1_B03_"
                "T0001F002L01A01Z01C01.png"
            ),
            target_folder / target_files[1],
        )
    return str(target_folder)

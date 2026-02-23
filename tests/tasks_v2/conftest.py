import os
import shutil
from pathlib import Path

import pytest
from ngio.utils import download_ome_zarr_dataset


@pytest.fixture(scope="session")
def zenodo_download_dir(testdata_path) -> Path:
    """
    Fixture to download the Zenodo dataset.
    """
    zenodo_download_dir = testdata_path / "ngio_zenodo"
    os.makedirs(zenodo_download_dir, exist_ok=True)
    return zenodo_download_dir


@pytest.fixture(scope="session")
def cardiomyocyte_tiny_source_path(zenodo_download_dir: Path) -> Path:
    """
    Fixture to download the CardiomyocyteTiny dataset from Zenodo.
    """
    return download_ome_zarr_dataset(
        "CardiomyocyteTiny", download_dir=zenodo_download_dir
    )


@pytest.fixture(scope="session")
def cardiomyocyte_small_mip_source_path(zenodo_download_dir: Path) -> Path:
    """
    Fixture to download the CardiomyocyteSmallMip dataset from Zenodo.
    """
    return download_ome_zarr_dataset(
        "CardiomyocyteSmallMip", download_dir=zenodo_download_dir
    )



@pytest.fixture
def cardiomyocyte_tiny_path(
    tmp_path: Path, cardiomyocyte_tiny_source_path: Path
) -> Path:
    dest_path = tmp_path / cardiomyocyte_tiny_source_path.stem
    shutil.copytree(cardiomyocyte_tiny_source_path, dest_path, dirs_exist_ok=True)
    return dest_path


@pytest.fixture
def cardiomyocyte_small_mip_path(
    tmp_path: Path, cardiomyocyte_small_mip_source_path: Path
) -> Path:
    dest_path = tmp_path / cardiomyocyte_small_mip_source_path.stem
    shutil.copytree(cardiomyocyte_small_mip_source_path, dest_path, dirs_exist_ok=True)
    return dest_path

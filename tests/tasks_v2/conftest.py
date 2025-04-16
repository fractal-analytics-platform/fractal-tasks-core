import os
import shutil
from pathlib import Path

import pytest
from ngio import create_empty_ome_zarr
from ngio import create_empty_plate
from ngio import ImageInWellPath
from ngio import OmeZarrPlate
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


def build_2w_1a_plate(plate_path: Path) -> OmeZarrPlate:
    """
    Build a plate with 2 wells and 1 acquisition.
    """

    images = [
        ImageInWellPath(row="A", column="01", path="0"),
        ImageInWellPath(row="B", column="02", path="0"),
    ]

    store = plate_path / "plate_xy_2w_1a.zarr"

    plate = create_empty_plate(
        store=store,
        name="plate_xy_2w_1a",
        images=images,
        overwrite=True,
        cache=True,
        parallel_safe=False,
    )
    return plate


def build_1w_2a_plate(plate_path: Path) -> OmeZarrPlate:
    """
    Build a plate with 1 well and 2 acquisitions.
    """

    images = [
        ImageInWellPath(row="A", column="01", path="0", acquisition_id=0),
        ImageInWellPath(row="A", column="01", path="1", acquisition_id=1),
    ]

    store = plate_path / "plate_xy_1w_2a.zarr"

    plate = create_empty_plate(
        store=store,
        name="plate_xy_1w_2a",
        images=images,
        overwrite=True,
        cache=True,
        parallel_safe=False,
    )
    return plate


def add_images_to_plate(
    plate: OmeZarrPlate, shape: tuple[int, ...], axes: str = "czyx"
) -> OmeZarrPlate:
    """
    Add images to the plate.
    """

    for image_path in plate.images_paths():
        row, column, path = image_path.split("/")
        ome_zarr = create_empty_ome_zarr(
            store=plate.get_image_store(row, column, path),
            shape=shape,
            xy_pixelsize=0.5,
            overwrite=True,
            levels=2,
            axes_names=axes,
        )
        table = ome_zarr.build_image_roi_table("image")
        ome_zarr.add_table(
            "well_ROI_table", table, backend="experimental_json_v1"
        )
    return plate


def _sample_plate_zarr_urls(
    tmp_path: Path,
    shape: tuple[int, ...],
    axes: str,
    plate_type: str = "2w_1a",
) -> list[str]:
    """
    Build a sample plate with different wells and acquisitions.
    """

    # Create a plate with 2 wells and 1 acquisition
    if plate_type == "2w_1a":
        plate = build_2w_1a_plate(tmp_path)
    elif plate_type == "1w_2a":
        plate = build_1w_2a_plate(tmp_path)

    else:
        raise ValueError(f"Unknown plate type: {plate_type}")

    # Add images to the plate
    plate = add_images_to_plate(plate, shape, axes)

    zarr_urls = []
    for image_path in plate.images_paths():
        base_url = plate._group_handler.store
        assert isinstance(base_url, Path)
        zarr_urls.append(str(base_url / image_path))
    return zarr_urls


def plate_2w_1a_yx(tmp_path: Path) -> list[str]:
    """
    Build a plate with 2 wells and 1 acquisition.
    """

    return _sample_plate_zarr_urls(
        tmp_path, shape=(10, 10), axes="yx", plate_type="2w_1a"
    )


def plate_1w_2a_c1yx(tmp_path: Path) -> list[str]:
    """
    Build a plate with 1 well and 2 acquisitions.
    """

    return _sample_plate_zarr_urls(
        tmp_path, shape=(3, 1, 10, 10), axes="czyx", plate_type="1w_2a"
    )


def plate_2w_1a_czyx(tmp_path: Path) -> list[str]:
    """
    Build a plate with 2 wells and 1 acquisition.
    """

    return _sample_plate_zarr_urls(
        tmp_path, shape=(3, 4, 10, 10), axes="czyx", plate_type="2w_1a"
    )


def plate_2w_1a_zyx(tmp_path: Path) -> list[str]:
    """
    Build a plate with 1 well and 2 acquisitions.
    """

    return _sample_plate_zarr_urls(
        tmp_path, shape=(4, 10, 10), axes="zyx", plate_type="2w_1a"
    )


def plate_2w_1a_tczyx(tmp_path: Path) -> list[str]:
    """
    Build a plate with 1 well and 2 acquisitions.
    """

    return _sample_plate_zarr_urls(
        tmp_path, shape=(2, 3, 4, 10, 10), axes="tczyx", plate_type="2w_1a"
    )


def plate_1w_2a_czyx(tmp_path: Path) -> list[str]:
    """
    Build a plate with 1 well and 2 acquisitions.
    """

    return _sample_plate_zarr_urls(
        tmp_path, shape=(3, 4, 10, 10), axes="czyx", plate_type="1w_2a"
    )


@pytest.fixture(scope="session")
def sample_ome_zarr_zyx_url(testdata_path) -> Path:
    """
    Fixture to create a sample OME-Zarr store.
    """
    store = testdata_path / "ngio_synt" / "sample_ome_zarr_zyx.zarr"
    origin_ome_zarr = create_empty_ome_zarr(
        store=store,
        shape=(16, 32, 32),
        xy_pixelsize=0.1,
        z_spacing=0.5,
        overwrite=True,
        axes_names="zyx",
    )
    table = origin_ome_zarr.build_image_roi_table("image")
    origin_ome_zarr.add_table(
        "well_ROI_table", table, backend="experimental_json_v1"
    )
    return store


@pytest.fixture
def cardiomyocyte_tiny_path(
    tmp_path: Path, cardiomyocyte_tiny_source_path: Path
) -> Path:
    dest_path = tmp_path / cardiomyocyte_tiny_source_path.stem
    shutil.copytree(
        cardiomyocyte_tiny_source_path, dest_path, dirs_exist_ok=True
    )
    return dest_path


@pytest.fixture
def cardiomyocyte_small_mip_path(
    tmp_path: Path, cardiomyocyte_small_mip_source_path: Path
) -> Path:
    dest_path = tmp_path / cardiomyocyte_small_mip_source_path.stem
    shutil.copytree(
        cardiomyocyte_small_mip_source_path, dest_path, dirs_exist_ok=True
    )
    return dest_path

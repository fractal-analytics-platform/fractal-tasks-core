from pathlib import Path

import pytest
from conftest import plate_1w_2a_czyx
from conftest import plate_2w_1a_czyx
from conftest import plate_2w_1a_zyx
from devtools import debug
from ngio import open_ome_zarr_plate

from fractal_tasks_core.tasks.copy_ome_zarr_hcs_plate import (
    copy_ome_zarr_hcs_plate,
)


@pytest.mark.parametrize(
    "sample_plate_zarr_urls",
    [
        plate_2w_1a_zyx,
        plate_1w_2a_czyx,
    ],
    indirect=True,
)
def test_copy_hcs_plate(sample_plate_zarr_urls: list[str], tmp_path: Path):
    debug(sample_plate_zarr_urls)
    parallel_list = copy_ome_zarr_hcs_plate(
        zarr_urls=sample_plate_zarr_urls, zarr_dir=str(tmp_path)
    )

    image = parallel_list["parallelization_list"][0]
    debug(image)
    assert Path(image["zarr_url"]).exists()

    origin_url = image["init_args"]["origin_url"]
    assert origin_url in sample_plate_zarr_urls

    *source_plate_url, _, _, _ = origin_url.split("/")
    source_plate_url = "/".join(source_plate_url)
    source_plate = open_ome_zarr_plate(
        source_plate_url, parallel_safe=False, cache=True
    )

    *dest_plate_url, _, _, _ = image["zarr_url"].split("/")
    dest_plate_url = "/".join(dest_plate_url)
    dest_plate = open_ome_zarr_plate(
        dest_plate_url, parallel_safe=False, cache=True
    )

    assert source_plate.columns == dest_plate.columns
    assert source_plate.rows == dest_plate.rows
    assert source_plate.images_paths() == dest_plate.images_paths()
    assert source_plate.acquisition_ids == dest_plate.acquisition_ids


def test_flexibility_copy_hcs(tmp_path: Path):
    zarr_urls = plate_2w_1a_czyx(tmp_path)

    # Run in subsets
    subset_dir = str(tmp_path / "subset")
    # Subset 1
    parallel_list_1 = copy_ome_zarr_hcs_plate(
        zarr_urls=[zarr_urls[0]],
        zarr_dir=subset_dir,
        overwrite_images=True,
        re_initialize_plate=True,
    )
    # Subset 2
    _ = copy_ome_zarr_hcs_plate(
        zarr_urls=[zarr_urls[1]],
        zarr_dir=subset_dir,
        overwrite_images=False,
        re_initialize_plate=False,
    )

    zarr_url = parallel_list_1["parallelization_list"][0]["zarr_url"]
    *subset_plate_url, _, _, _ = zarr_url.split("/")
    subset_plate_url = "/".join(subset_plate_url)
    subset_plate = open_ome_zarr_plate(
        subset_plate_url, parallel_safe=False, cache=True
    )

    # Run all
    all_dir = str(tmp_path / "all")
    parallel_list_all = copy_ome_zarr_hcs_plate(
        zarr_urls=zarr_urls,
        zarr_dir=all_dir,
        overwrite_images=True,
        re_initialize_plate=True,
    )
    zarr_url = parallel_list_all["parallelization_list"][0]["zarr_url"]
    *all_plate_url, _, _, _ = zarr_url.split("/")
    all_plate_url = "/".join(all_plate_url)
    all_plate = open_ome_zarr_plate(
        all_plate_url, parallel_safe=False, cache=True
    )

    assert subset_plate.columns == all_plate.columns
    assert subset_plate.rows == all_plate.rows
    assert subset_plate.images_paths() == all_plate.images_paths()
    assert subset_plate.acquisition_ids == all_plate.acquisition_ids


def test_fail_overwrite(tmp_path: Path):
    zarr_urls = plate_2w_1a_czyx(tmp_path)

    copy_ome_zarr_hcs_plate(
        zarr_urls=zarr_urls,
        zarr_dir=str(tmp_path),
        overwrite_images=False,
        re_initialize_plate=False,
    )

    copy_ome_zarr_hcs_plate(
        zarr_urls=zarr_urls,
        zarr_dir=str(tmp_path),
        overwrite_images=True,
        re_initialize_plate=True,
    )

    with pytest.raises(FileExistsError):
        copy_ome_zarr_hcs_plate(
            zarr_urls=zarr_urls,
            zarr_dir=str(tmp_path),
            overwrite_images=False,
            re_initialize_plate=False,
        )


def test_reinit_true(tmp_path: Path):
    zarr_urls = plate_2w_1a_czyx(tmp_path)

    # Run in subsets
    # Subset 1
    parallel_list_1 = copy_ome_zarr_hcs_plate(
        zarr_urls=[zarr_urls[0]],
        zarr_dir=str(tmp_path),
        overwrite_images=True,
        re_initialize_plate=True,
    )

    zarr_url = parallel_list_1["parallelization_list"][0]["zarr_url"]
    assert Path(zarr_url).exists()

    # Subset 2 will reinitialize the plate
    # and overwrite the images
    parallel_list_2 = copy_ome_zarr_hcs_plate(
        zarr_urls=[zarr_urls[1]],
        zarr_dir=str(tmp_path),
        overwrite_images=False,
        re_initialize_plate=True,
    )

    zarr_url = parallel_list_2["parallelization_list"][0]["zarr_url"]
    assert Path(zarr_url).exists()

    zarr_url = parallel_list_1["parallelization_list"][0]["zarr_url"]
    assert not Path(zarr_url).exists(), zarr_url

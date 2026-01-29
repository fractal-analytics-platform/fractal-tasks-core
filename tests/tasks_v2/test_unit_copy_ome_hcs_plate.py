from pathlib import Path
from typing import Callable

import pytest
from conftest import plate_1w_2a_czyx
from conftest import plate_2w_1a_czyx
from conftest import plate_2w_1a_zyx
from devtools import debug
from ngio import OmeZarrPlate
from ngio import open_ome_zarr_plate

from fractal_tasks_core.tasks.copy_ome_zarr_hcs_plate import (
    copy_ome_zarr_hcs_plate,
)


def _get_plate(zarr_url: str) -> OmeZarrPlate:
    """
    Get the plate from the parallelization list.
    """
    *plate_url, _, _, _ = zarr_url.split("/")
    plate_url = "/".join(plate_url)
    return open_ome_zarr_plate(plate_url, parallel_safe=False, cache=True)


@pytest.mark.parametrize(
    "create_plate",
    [
        plate_2w_1a_zyx,
        plate_1w_2a_czyx,
    ],
)
def test_copy_hcs_plate(create_plate: Callable, tmp_path: Path):
    # Create a sample plate and returns a list of zarr urls
    sample_plate_zarr_urls = create_plate(tmp_path)
    debug(sample_plate_zarr_urls)
    parallel_list = copy_ome_zarr_hcs_plate(
        zarr_urls=sample_plate_zarr_urls, zarr_dir=str(tmp_path)
    )

    image = parallel_list["parallelization_list"][0]
    debug(image)
    assert Path(image["zarr_url"]).parent.exists()

    origin_url = image["init_args"]["origin_url"]
    assert origin_url in sample_plate_zarr_urls

    source_plate = _get_plate(origin_url)
    dest_plate = _get_plate(image["zarr_url"])

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
        overwrite=True,
        re_initialize_plate=True,
    )
    # Subset 2
    _ = copy_ome_zarr_hcs_plate(
        zarr_urls=[zarr_urls[1]],
        zarr_dir=subset_dir,
        overwrite=False,
        re_initialize_plate=False,
    )

    zarr_url = parallel_list_1["parallelization_list"][0]["zarr_url"]
    subset_plate = _get_plate(zarr_url)

    # Run all
    all_dir = str(tmp_path / "all")
    parallel_list_all = copy_ome_zarr_hcs_plate(
        zarr_urls=zarr_urls,
        zarr_dir=all_dir,
        overwrite=True,
        re_initialize_plate=True,
    )
    zarr_url = parallel_list_all["parallelization_list"][0]["zarr_url"]
    all_plate = _get_plate(zarr_url)

    assert subset_plate.columns == all_plate.columns
    assert subset_plate.rows == all_plate.rows
    assert subset_plate.images_paths() == all_plate.images_paths()
    assert subset_plate.acquisition_ids == all_plate.acquisition_ids


def test_fail_overwrite(tmp_path: Path):
    zarr_urls = plate_2w_1a_czyx(tmp_path)

    copy_ome_zarr_hcs_plate(
        zarr_urls=zarr_urls,
        zarr_dir=str(tmp_path),
        overwrite=False,
        re_initialize_plate=False,
    )

    # Run with re_initialize_plate=True
    # Should remove all existing images
    copy_ome_zarr_hcs_plate(
        zarr_urls=zarr_urls,
        zarr_dir=str(tmp_path),
        overwrite=False,
        re_initialize_plate=True,
    )

    with pytest.raises(FileExistsError):
        copy_ome_zarr_hcs_plate(
            zarr_urls=zarr_urls,
            zarr_dir=str(tmp_path),
            overwrite=False,
            re_initialize_plate=False,
        )


def test_new_plate_name(tmp_path: Path):
    zarr_urls = plate_2w_1a_czyx(tmp_path)

    for projection_axis in ["x", "y", "z"]:
        axis_suffix = "" if projection_axis == "z" else f"_{projection_axis}"
        for method in ["mip", "minip", "meanip", "sumip"]:
            p_list = copy_ome_zarr_hcs_plate(
                zarr_urls=[zarr_urls[0]],
                zarr_dir=str(tmp_path),
                method=method,
                advanced_parameters={"projection_axis": projection_axis},
                overwrite=False,
                re_initialize_plate=False,
            )

            test_new_plate_name = p_list["parallelization_list"][0][
                "init_args"
            ]["new_plate_name"]
            assert (
                test_new_plate_name
                == f"plate_xy_2w_1_{method}{axis_suffix}.zarr"
            )


def test_fail_not_plate_url():
    with pytest.raises(ValueError):
        # Test with a non-image-in-plate URL
        copy_ome_zarr_hcs_plate(
            zarr_urls=["/tmp/plate.zarr"],
            zarr_dir="/tmp",
            overwrite=False,
            re_initialize_plate=False,
        )


def test_reinit_true(tmp_path: Path):
    zarr_urls = plate_2w_1a_czyx(tmp_path)

    # Run in subsets
    # Subset 1
    parallel_list_1 = copy_ome_zarr_hcs_plate(
        zarr_urls=[zarr_urls[0]],
        zarr_dir=str(tmp_path),
        overwrite=True,
        re_initialize_plate=True,
    )

    zarr_url = parallel_list_1["parallelization_list"][0]["zarr_url"]
    assert Path(zarr_url).parent.exists()

    # Subset 2 will reinitialize the plate
    # and overwrite the images
    parallel_list_2 = copy_ome_zarr_hcs_plate(
        zarr_urls=[zarr_urls[1]],
        zarr_dir=str(tmp_path),
        overwrite=False,
        re_initialize_plate=True,
    )

    zarr_url = parallel_list_2["parallelization_list"][0]["zarr_url"]
    assert Path(zarr_url).parent.exists()

    zarr_url = parallel_list_1["parallelization_list"][0]["zarr_url"]
    assert not Path(zarr_url).parent.exists(), zarr_url

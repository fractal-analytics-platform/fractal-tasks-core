import shutil
from pathlib import Path

import pytest
from ngio import open_ome_zarr_container

from fractal_tasks_core.tasks.illumination_correction import (
    illumination_correction,
)


def _check_that_images_differs(first_url: str, second_url: str) -> None:
    """
    Compare that the images differ and they are not empty.
    """
    first_ome_zarr = open_ome_zarr_container(first_url)
    first_image = first_ome_zarr.get_image()
    first_data = first_image.get_array()

    second_ome_zarr = open_ome_zarr_container(second_url)
    second_image = second_ome_zarr.get_image()
    second_data = second_image.get_array()

    assert first_data.any(), "First image is empty"
    assert second_data.any(), "Second image is empty"
    assert not (second_data == first_data).all(), "Images are identical"


@pytest.mark.parametrize("overwrite_input", [True, False])
def test_output_handled(
    cardiomyocyte_small_mip_path: Path,
    tmp_path: Path,
    testdata_path: Path,
    overwrite_input: bool,
) -> None:
    image_url = str(cardiomyocyte_small_mip_path / "B" / "03" / "0")

    # copy input data for comparison
    saved_origin_url: str = str(tmp_path / "input_image.zarr")
    shutil.copytree(image_url, saved_origin_url)

    # Prepare arguments for illumination_correction function
    testdata_str = testdata_path.as_posix()

    illumination_profiles_folder: str = (
        f"{testdata_str}/illumination_correction/"
    )
    illumination_profiles_map: dict[str, str] = {
        "A01_C01": "flatfield_corr_matrix.png",
        "A01_C02": "flatfield_corr_matrix.png",
        "A02_C03": "flatfield_corr_matrix.png",
    }
    illumination_profiles = {
        "folder": illumination_profiles_folder,
        "profiles": illumination_profiles_map,
    }

    # do illumination correction
    illumination_correction(
        zarr_url=image_url,
        illumination_profiles=illumination_profiles,
        overwrite_input=overwrite_input,
        suffix="_corrected",
    )

    # check output paths
    new_image_url = image_url + "_corrected"
    # origin exists always
    assert Path(image_url).exists()
    # _corrected image should exist only if not overwriting
    assert Path(new_image_url).exists() != overwrite_input

    # corrected image should be non-empty and different from origin
    output_image_url = image_url if overwrite_input else new_image_url
    _check_that_images_differs(saved_origin_url, output_image_url)


@pytest.mark.parametrize(
    (
        "illumination_profiles_map",
        "background_profiles_map",
        "problematic_wavelength",
    ),
    [
        # only illumination profiles, one wavelength missing
        (
            {
                "A01_C01": "flatfield_corr_matrix.png",
                "A01_C02": "flatfield_corr_matrix.png",
            },
            {},
            "A02_C03",
        ),
        # too much illumination profiles, one extra wavelength
        (
            {
                "A00_C00": "flatfield_corr_matrix.png",
                "A01_C01": "flatfield_corr_matrix.png",
                "A01_C02": "flatfield_corr_matrix.png",
                "A02_C03": "flatfield_corr_matrix.png",
            },
            {},
            "A00_C00",
        ),
        # all illumination profiles, missing background for one wavelength
        (
            {
                "A01_C01": "flatfield_corr_matrix.png",
                "A01_C02": "flatfield_corr_matrix.png",
                "A02_C03": "flatfield_corr_matrix.png",
            },
            {
                "A01_C01": "darkfield_corr_matrix.png",
                "A02_C03": "darkfield_corr_matrix.png",
            },
            "A01_C02",
        ),
        # all background profiles, missing illumination for one wavelength
        (
            {
                "A01_C01": "flatfield_corr_matrix.png",
                "A02_C03": "flatfield_corr_matrix.png",
            },
            {
                "A01_C01": "darkfield_corr_matrix.png",
                "A01_C02": "darkfield_corr_matrix.png",
                "A02_C03": "darkfield_corr_matrix.png",
            },
            "A01_C02",
        ),
        # no profiles at all
        ({}, {}, "A01_C01"),
    ],
)
def test_wrong_wavelength_profiles(
    cardiomyocyte_small_mip_path: Path,
    testdata_path: Path,
    illumination_profiles_map: dict[str, str],
    background_profiles_map: dict[str, str],
    problematic_wavelength: str,
) -> None:
    image_url = str(cardiomyocyte_small_mip_path / "B" / "03" / "0")

    # Prepare arguments for illumination_correction function
    testdata_str = testdata_path.as_posix()

    illumination_profiles_folder: str = (
        f"{testdata_str}/illumination_correction/"
    )
    illumination_profiles = {
        "folder": illumination_profiles_folder,
        "profiles": illumination_profiles_map,
    }
    background_profiles_folder: str = (
        f"{testdata_str}/illumination_correction/"
    )
    background_profiles = {
        "folder": background_profiles_folder,
        "profiles": background_profiles_map,
        "model": "Profile",
    }

    # do illumination correction
    with pytest.raises(
        ValueError,
        match=problematic_wavelength,
    ):
        illumination_correction(
            zarr_url=image_url,
            illumination_profiles=illumination_profiles,
            background_correction={"value": background_profiles},
            overwrite_input=False,
            suffix="_corrected",
        )


def test_constant_background_subtraction(
    cardiomyocyte_small_mip_path: Path,
    testdata_path: Path,
) -> None:
    image_url = str(cardiomyocyte_small_mip_path / "B" / "03" / "0")

    # Prepare arguments for illumination_correction function
    testdata_str = testdata_path.as_posix()

    illumination_profiles_folder: str = (
        f"{testdata_str}/illumination_correction/"
    )
    illumination_profiles_map: dict[str, str] = {
        "A01_C01": "flatfield_corr_matrix.png",
        "A01_C02": "flatfield_corr_matrix.png",
        "A02_C03": "flatfield_corr_matrix.png",
    }
    illumination_profiles = {
        "folder": illumination_profiles_folder,
        "profiles": illumination_profiles_map,
    }

    background_model = {
        "value": {
            "model": "Constant",
            "constants": {
                "A01_C01": 100,
                "A01_C02": 150,
                "A02_C03": 200,
            },
        },
    }

    # do illumination correction
    illumination_correction(
        zarr_url=image_url,
        illumination_profiles=illumination_profiles,
        background_correction=background_model,
        overwrite_input=False,
        suffix="_with_background",
    )
    illumination_correction(
        zarr_url=image_url,
        illumination_profiles=illumination_profiles,
        overwrite_input=False,
        suffix="_no_background",
    )

    # corrected with background profiles should differ from corrected without
    _check_that_images_differs(
        image_url + "_with_background", image_url + "_no_background"
    )


def test_with_background_profiles(
    cardiomyocyte_small_mip_path: Path,
    testdata_path: Path,
) -> None:
    image_url = str(cardiomyocyte_small_mip_path / "B" / "03" / "0")

    # Prepare arguments for illumination_correction function
    testdata_str = testdata_path.as_posix()

    illumination_profiles_folder: str = (
        f"{testdata_str}/illumination_correction/"
    )
    illumination_profiles_map: dict[str, str] = {
        "A01_C01": "flatfield_corr_matrix.png",
        "A01_C02": "flatfield_corr_matrix.png",
        "A02_C03": "flatfield_corr_matrix.png",
    }
    background_profiles_folder: str = (
        f"{testdata_str}/illumination_correction/"
    )
    background_profiles_map: dict[str, str] = {
        "A01_C01": "darkfield_corr_matrix.png",
        "A01_C02": "darkfield_corr_matrix.png",
        "A02_C03": "darkfield_corr_matrix.png",
    }
    illumination_profiles = {
        "folder": illumination_profiles_folder,
        "profiles": illumination_profiles_map,
    }
    background_profiles = {
        "folder": background_profiles_folder,
        "profiles": background_profiles_map,
        "model": "Profile",
    }
    background_correction = {
        "value": background_profiles,
    }

    # do illumination correction
    illumination_correction(
        zarr_url=image_url,
        illumination_profiles=illumination_profiles,
        background_correction=background_correction,
        overwrite_input=False,
        suffix="_with_background",
    )

    # control group with no background profiles
    illumination_correction(
        zarr_url=image_url,
        illumination_profiles=illumination_profiles,
        overwrite_input=False,
        suffix="_no_background",
    )

    _check_that_images_differs(
        image_url + "_no_background", image_url + "_with_background"
    )


def test_two_different_illumination_profiles(
    cardiomyocyte_small_mip_path: Path,
    testdata_path: Path,
) -> None:
    # Check if providing different files gives different results
    image_url = str(cardiomyocyte_small_mip_path / "B" / "03" / "0")

    # Prepare arguments for illumination_correction function
    testdata_str = testdata_path.as_posix()

    illumination_profiles_folder: str = (
        f"{testdata_str}/illumination_correction/"
    )
    illumination_profiles_map: dict[str, str] = {
        "A01_C01": "flatfield_corr_matrix.png",
        "A01_C02": "illum_corr_matrix.png",
        "A02_C03": "flatfield_corr_matrix.png",
    }
    illumination_profiles = {
        "folder": illumination_profiles_folder,
        "profiles": illumination_profiles_map,
    }

    # do illumination correction
    illumination_correction(
        zarr_url=image_url,
        illumination_profiles=illumination_profiles,
        overwrite_input=False,
        suffix="_corrected2",
    )

    # control group
    illumination_profiles_control_map: dict[str, str] = {
        "A01_C01": "flatfield_corr_matrix.png",
        "A01_C02": "flatfield_corr_matrix.png",
        "A02_C03": "flatfield_corr_matrix.png",
    }
    illumination_profiles_control = {
        "folder": illumination_profiles_folder,
        "profiles": illumination_profiles_control_map,
    }
    illumination_correction(
        zarr_url=image_url,
        illumination_profiles=illumination_profiles_control,
        overwrite_input=False,
        suffix="_corrected",
    )

    _check_that_images_differs(
        image_url + "_corrected", image_url + "_corrected2"
    )


def test_wrong_file_or_folder(
    cardiomyocyte_small_mip_path: Path,
    testdata_path: Path,
) -> None:
    image_url = str(cardiomyocyte_small_mip_path / "B" / "03" / "0")

    # Prepare arguments for illumination_correction function
    testdata_str = testdata_path.as_posix()

    wrong_folder = f"{testdata_str}/non_existing_folder/"

    illumination_profiles_folder: str = (
        f"{testdata_str}/illumination_correction/"
    )
    illumination_profiles_map: dict[str, str] = {
        "A01_C01": "flatfield_corr_matrix.png",
        "A01_C02": "flatfield_corr_matrix.png",
        "A02_C03": "flatfield_corr_matrix.png",
    }
    illumination_profiles = {
        "folder": illumination_profiles_folder,
        "profiles": illumination_profiles_map,
    }
    background_profiles_folder: str = illumination_profiles_folder
    background_profiles_map: dict[str, str] = {
        "A01_C01": "darkfield_corr_matrix.png",
        "A01_C02": "darkfield_corr_matrix.png",
        "A02_C03": "darkfield_corr_matrix.png",
    }
    background_profiles = {
        "folder": background_profiles_folder,
        "profiles": background_profiles_map,
        "model": "Profile",
    }

    # test illumination folder wrong
    illumination_profiles_wrong_folder = {
        "folder": wrong_folder,
        "profiles": illumination_profiles_map,
    }
    with pytest.raises(FileNotFoundError, match="No such file"):
        illumination_correction(
            zarr_url=image_url,
            illumination_profiles=illumination_profiles_wrong_folder,
            background_correction={"value": background_profiles},
            overwrite_input=False,
            suffix="_corrected",
        )

    # test background folder wrong
    background_profiles_wrong_folder = {
        "folder": wrong_folder,
        "profiles": background_profiles_map,
        "model": "Profile",
    }
    with pytest.raises(FileNotFoundError, match="No such file"):
        illumination_correction(
            zarr_url=image_url,
            illumination_profiles=illumination_profiles,
            background_correction={"value": background_profiles_wrong_folder},
            overwrite_input=False,
            suffix="_corrected",
        )

    # test illumination file wrong
    illumination_profiles_map_wrong: dict[str, str] = {
        "A01_C01": "non_existing_file.png",
        "A01_C02": "flatfield_corr_matrix.png",
        "A02_C03": "flatfield_corr_matrix.png",
    }
    illumination_profiles_wrong = {
        "folder": illumination_profiles_folder,
        "profiles": illumination_profiles_map_wrong,
    }
    with pytest.raises(FileNotFoundError, match="No such file"):
        illumination_correction(
            zarr_url=image_url,
            illumination_profiles=illumination_profiles_wrong,
            background_correction={"value": background_profiles},
            overwrite_input=False,
            suffix="_corrected",
        )

    # test background file wrong
    background_profiles_map_wrong: dict[str, str] = {
        "A01_C01": "darkfield_corr_matrix.png",
        "A01_C02": "non_existing_file.png",
        "A02_C03": "darkfield_corr_matrix.png",
    }
    background_profiles_wrong = {
        "folder": background_profiles_folder,
        "profiles": background_profiles_map_wrong,
        "model": "Profile",
    }
    with pytest.raises(FileNotFoundError, match="No such file"):
        illumination_correction(
            zarr_url=image_url,
            illumination_profiles=illumination_profiles,
            background_correction={"value": background_profiles_wrong},
            overwrite_input=False,
            suffix="_corrected",
        )

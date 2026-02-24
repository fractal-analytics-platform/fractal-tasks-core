import shutil
from pathlib import Path

import numpy as np
import pytest
from ngio import Roi, create_empty_ome_zarr, open_ome_zarr_container
from ngio.tables import RoiTable
from skimage.io import imsave

from fractal_tasks_core._io_models import ConstantCorrectionModel
from fractal_tasks_core.illumination_correction import (
    BackgroundCorrection,
    ProfileCorrectionModel,
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

    illumination_profiles_folder: str = f"{testdata_str}/illumination_correction/"
    illumination_profiles_map: dict[str, str] = {
        "A01_C01": "flatfield_corr_matrix.png",
        "A01_C02": "flatfield_corr_matrix.png",
        "A02_C03": "flatfield_corr_matrix.png",
    }
    illumination_profiles = ProfileCorrectionModel(
        folder=illumination_profiles_folder,
        profiles=illumination_profiles_map,
    )

    # do illumination correction
    task_update_list = illumination_correction(
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

    if overwrite_input:
        assert task_update_list is None
    else:
        assert task_update_list is not None
        assert len(task_update_list["image_list_updates"]) == 1
        assert task_update_list["image_list_updates"][0]["zarr_url"] == new_image_url
        assert task_update_list["image_list_updates"][0]["origin"] == image_url


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

    illumination_profiles_folder: str = f"{testdata_str}/illumination_correction/"
    illumination_profiles = ProfileCorrectionModel(
        folder=illumination_profiles_folder,
        profiles=illumination_profiles_map,
    )
    background_profiles_folder: str = f"{testdata_str}/illumination_correction/"
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
            background_correction={"value": background_profiles},  # type: ignore wrong profile
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

    illumination_profiles_folder: str = f"{testdata_str}/illumination_correction/"
    illumination_profiles_map: dict[str, str] = {
        "A01_C01": "flatfield_corr_matrix.png",
        "A01_C02": "flatfield_corr_matrix.png",
        "A02_C03": "flatfield_corr_matrix.png",
    }
    illumination_profiles = ProfileCorrectionModel(
        folder=illumination_profiles_folder,
        profiles=illumination_profiles_map,
    )

    background_model = BackgroundCorrection(
        value=ConstantCorrectionModel(
            constants={
                "A01_C01": 100,
                "A01_C02": 150,
                "A02_C03": 200,
            },
        ),
    )

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

    illumination_profiles_folder: str = f"{testdata_str}/illumination_correction/"
    illumination_profiles_map: dict[str, str] = {
        "A01_C01": "flatfield_corr_matrix.png",
        "A01_C02": "flatfield_corr_matrix.png",
        "A02_C03": "flatfield_corr_matrix.png",
    }
    background_profiles_folder: str = f"{testdata_str}/illumination_correction/"
    background_profiles_map: dict[str, str] = {
        "A01_C01": "darkfield_corr_matrix.png",
        "A01_C02": "darkfield_corr_matrix.png",
        "A02_C03": "darkfield_corr_matrix.png",
    }
    illumination_profiles = ProfileCorrectionModel(
        folder=illumination_profiles_folder,
        profiles=illumination_profiles_map,
    )
    background_correction = BackgroundCorrection(
        value=ProfileCorrectionModel(
            folder=background_profiles_folder,
            profiles=background_profiles_map,
        )
    )

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

    illumination_profiles_folder: str = f"{testdata_str}/illumination_correction/"
    illumination_profiles_map: dict[str, str] = {
        "A01_C01": "flatfield_corr_matrix.png",
        "A01_C02": "illum_corr_matrix.png",
        "A02_C03": "flatfield_corr_matrix.png",
    }
    illumination_profiles = ProfileCorrectionModel(
        folder=illumination_profiles_folder,
        profiles=illumination_profiles_map,
    )

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
    illumination_profiles_control = ProfileCorrectionModel(
        folder=illumination_profiles_folder,
        profiles=illumination_profiles_control_map,
    )
    illumination_correction(
        zarr_url=image_url,
        illumination_profiles=illumination_profiles_control,
        overwrite_input=False,
        suffix="_corrected",
    )

    _check_that_images_differs(image_url + "_corrected", image_url + "_corrected2")


def test_wrong_file_or_folder(
    cardiomyocyte_small_mip_path: Path,
    testdata_path: Path,
) -> None:
    image_url = str(cardiomyocyte_small_mip_path / "B" / "03" / "0")

    # Prepare arguments for illumination_correction function
    testdata_str = testdata_path.as_posix()

    wrong_folder = f"{testdata_str}/non_existing_folder/"

    illumination_profiles_folder: str = f"{testdata_str}/illumination_correction/"
    illumination_profiles_map: dict[str, str] = {
        "A01_C01": "flatfield_corr_matrix.png",
        "A01_C02": "flatfield_corr_matrix.png",
        "A02_C03": "flatfield_corr_matrix.png",
    }
    illumination_profiles = ProfileCorrectionModel(
        folder=illumination_profiles_folder,
        profiles=illumination_profiles_map,
    )
    background_profiles_folder: str = illumination_profiles_folder
    background_profiles_map: dict[str, str] = {
        "A01_C01": "darkfield_corr_matrix.png",
        "A01_C02": "darkfield_corr_matrix.png",
        "A02_C03": "darkfield_corr_matrix.png",
    }
    background_profiles = ProfileCorrectionModel(
        folder=background_profiles_folder,
        profiles=background_profiles_map,
        model="Profile",
    )
    background_correction = BackgroundCorrection(value=background_profiles)

    # test illumination folder wrong
    illumination_profiles_wrong_folder = ProfileCorrectionModel(
        folder=wrong_folder,
        profiles=illumination_profiles_map,
    )
    with pytest.raises(FileNotFoundError, match="No such file"):
        illumination_correction(
            zarr_url=image_url,
            illumination_profiles=illumination_profiles_wrong_folder,
            background_correction=background_correction,
            overwrite_input=False,
            suffix="_corrected",
        )

    # test background folder wrong
    background_profiles_wrong_folder = ProfileCorrectionModel(
        folder=wrong_folder,
        profiles=background_profiles_map,
        model="Profile",
    )
    background_correction_wrong_folder = BackgroundCorrection(
        value=background_profiles_wrong_folder
    )
    with pytest.raises(FileNotFoundError, match="No such file"):
        illumination_correction(
            zarr_url=image_url,
            illumination_profiles=illumination_profiles,
            background_correction=background_correction_wrong_folder,
            overwrite_input=False,
            suffix="_corrected",
        )

    # test illumination file wrong
    illumination_profiles_map_wrong: dict[str, str] = {
        "A01_C01": "non_existing_file.png",
        "A01_C02": "flatfield_corr_matrix.png",
        "A02_C03": "flatfield_corr_matrix.png",
    }
    illumination_profiles_wrong = ProfileCorrectionModel(
        folder=illumination_profiles_folder,
        profiles=illumination_profiles_map_wrong,
    )
    with pytest.raises(FileNotFoundError, match="No such file"):
        illumination_correction(
            zarr_url=image_url,
            illumination_profiles=illumination_profiles_wrong,
            background_correction=background_correction,
            overwrite_input=False,
            suffix="_corrected",
        )

    # test background file wrong
    background_profiles_map_wrong: dict[str, str] = {
        "A01_C01": "darkfield_corr_matrix.png",
        "A01_C02": "non_existing_file.png",
        "A02_C03": "darkfield_corr_matrix.png",
    }
    background_profiles_wrong = ProfileCorrectionModel(
        folder=background_profiles_folder,
        profiles=background_profiles_map_wrong,
        model="Profile",
    )
    background_correction_wrong = BackgroundCorrection(value=background_profiles_wrong)
    with pytest.raises(FileNotFoundError, match="No such file"):
        illumination_correction(
            zarr_url=image_url,
            illumination_profiles=illumination_profiles,
            background_correction=background_correction_wrong,
            overwrite_input=False,
            suffix="_corrected",
        )


@pytest.mark.parametrize(
    "shape, axes",
    [
        ((2160, 2560), "yx"),
        ((16, 2160, 2560), "zyx"),
        ((3, 16, 2160, 2560), "czyx"),
        ((4, 3, 16, 2160, 2560), "tczyx"),
    ],
)
def test_multidimensional_input(
    shape, axes: str, tmp_path: Path, testdata_path: Path
) -> None:
    """
    Test the projection task.
    """
    store = tmp_path / "sample_ome_zarr.zarr"
    print(store)
    origin_ome_zarr = create_empty_ome_zarr(
        store=store.as_posix(),
        shape=shape,
        pixelsize=0.1,
        z_spacing=0.5,
        overwrite=False,
        axes_names=axes,
    )

    table = origin_ome_zarr.build_image_roi_table("image")
    origin_ome_zarr.add_table("well_ROI_table", table, backend="anndata")

    # Prepare arguments for illumination_correction function
    testdata_str = testdata_path.as_posix()

    illumination_profiles_folder: str = f"{testdata_str}/illumination_correction/"
    illumination_profiles_map: dict[str, str] = {}
    for wavelength in origin_ome_zarr.wavelength_ids:
        assert wavelength is not None
        illumination_profiles_map[wavelength] = "flatfield_corr_matrix.png"
    illumination_profiles = ProfileCorrectionModel(
        folder=illumination_profiles_folder,
        profiles=illumination_profiles_map,
    )

    print(store)
    print(store.as_posix())
    illumination_correction(
        zarr_url=store.as_posix(),
        illumination_profiles=illumination_profiles,
        overwrite_input=False,
        input_ROI_table="well_ROI_table",
    )

    # init_mip = InitArgsMIP(
    #     origin_url=str(store),
    #     method="mip",
    #     overwrite=False,
    #     new_plate_name="new_plate.zarr",
    # )

    # mip_store = tmp_path / "sample_ome_zarr_mip.zarr"
    # update_list = projection(zarr_url=str(mip_store), init_args=init_mip)

    # zarr_url = update_list["image_list_updates"][0]["zarr_url"]
    # origin_url = update_list["image_list_updates"][0]["origin"]
    # attributes = update_list["image_list_updates"][0]["attributes"]
    # types = update_list["image_list_updates"][0]["types"]


# ---------------------------------------------------------------------------
# empty suffix raises before any zarr I/O
# ---------------------------------------------------------------------------


def test_empty_suffix() -> None:
    illumination_profile = ProfileCorrectionModel(
        folder="/any",
        profiles={},
    )
    with pytest.raises(ValueError, match="suffix cannot be an empty string"):
        illumination_correction(
            zarr_url="/nonexistent",
            illumination_profiles=illumination_profile,
            overwrite_input=False,
            suffix="",
        )


# ---------------------------------------------------------------------------
# Constant background model wavelength mismatch
# ---------------------------------------------------------------------------


def _make_tiny_czyx_zarr(tmp_path: Path) -> tuple[str, list[str]]:
    """Create a minimal 2-channel czyx zarr; return (zarr_url, wavelength_ids)."""
    store = tmp_path / "tiny_czyx.zarr"
    ome_zarr = create_empty_ome_zarr(
        store=store,
        shape=(2, 1, 10, 10),
        pixelsize=0.5,
        z_spacing=1.0,
        axes_names="czyx",
        overwrite=True,
        levels=2,
    )

    wavelength_ids = [w for w in ome_zarr.wavelength_ids if w is not None]
    return str(store), wavelength_ids


@pytest.mark.parametrize(
    "constants_factory,match",
    [
        # extra wavelength in constants → first ValueError branch (lines 208-212)
        (
            lambda wl: {wl[0]: 100, wl[1]: 100, "channel_EXTRA": 50},
            "channel_EXTRA",
        ),
        # missing wavelength in constants → second ValueError branch (lines 214-218)
        (
            lambda wl: {wl[0]: 100},
            "channel_1",
        ),
    ],
)
def test_wrong_constant_background_wavelengths(
    tmp_path: Path,
    constants_factory,
    match: str,
) -> None:
    zarr_url, wavelengths = _make_tiny_czyx_zarr(tmp_path)

    illumination_profiles = ProfileCorrectionModel(
        folder="/fake",
        profiles={w: "fake.png" for w in wavelengths},
    )
    background_model = BackgroundCorrection(
        value=ConstantCorrectionModel(
            constants=constants_factory(wavelengths),
        ),
    )

    with pytest.raises(ValueError, match=match):
        illumination_correction(
            zarr_url=zarr_url,
            illumination_profiles=illumination_profiles,
            background_correction=background_model,
            overwrite_input=True,
        )


# ---------------------------------------------------------------------------
# Inconsistent FOV sizes in the ROI table
# ---------------------------------------------------------------------------


def test_inconsistent_fov_sizes(tmp_path: Path) -> None:
    store = tmp_path / "inconsistent.zarr"
    ome_zarr = create_empty_ome_zarr(
        store=store,
        shape=(1, 20, 20),
        pixelsize=0.5,
        z_spacing=1.0,
        axes_names="zyx",
        overwrite=True,
        levels=2,
    )
    # ROI 1: 5×5 µm → 10×10 px; ROI 2: 10×10 µm → 20×20 px — inconsistent sizes
    roi1 = Roi.from_values(
        name="fov_1", slices={"x": (0.0, 5.0), "y": (0.0, 5.0), "z": (0.0, 1.0)}
    )
    roi2 = Roi.from_values(
        name="fov_2", slices={"x": (5.0, 10.0), "y": (0.0, 10.0), "z": (0.0, 1.0)}
    )
    roi_table = RoiTable(rois=[roi1, roi2])
    ome_zarr.add_table("well_ROI_table", roi_table, backend="anndata")

    wavelengths = [w for w in ome_zarr.wavelength_ids if w is not None]
    illumination_profiles = ProfileCorrectionModel(
        folder="/fake",
        profiles={w: "fake.png" for w in wavelengths},
    )

    with pytest.raises(ValueError, match="Inconsistent image sizes"):
        illumination_correction(
            zarr_url=str(store),
            illumination_profiles=illumination_profiles,
            overwrite_input=True,
            input_ROI_table="well_ROI_table",
        )


# ---------------------------------------------------------------------------
# Correction matrix shape mismatch (flatfield and darkfield)
# ---------------------------------------------------------------------------


def _make_tiny_zyx_zarr_with_roi(tmp_path: Path) -> tuple[str, list[str]]:
    """Create a minimal 1-channel zyx zarr with a single-ROI well table."""
    store = tmp_path / "tiny_zyx.zarr"
    ome_zarr = create_empty_ome_zarr(
        store=store,
        shape=(1, 10, 10),
        pixelsize=0.5,
        z_spacing=1.0,
        axes_names="zyx",
        overwrite=True,
        levels=2,
    )
    table = ome_zarr.build_image_roi_table("image")
    ome_zarr.add_table("well_ROI_table", table, backend="anndata")
    wavelengths = list(w for w in ome_zarr.wavelength_ids if w is not None)
    return str(store), wavelengths


def _save_tiny_flatfield(path: Path) -> None:
    """Save a valid 10×10 flatfield PNG (uniform, no zeros) to *path*."""
    imsave(str(path), np.full((10, 10), 100, dtype=np.uint16))


def test_flatfield_shape_mismatch(tmp_path: Path, testdata_path: Path) -> None:
    zarr_url, wavelengths = _make_tiny_zyx_zarr_with_roi(tmp_path)
    # The existing PNG is 2160×2560; the image is 10×10 → mismatch
    illum_profiles = ProfileCorrectionModel(
        folder=str(testdata_path / "illumination_correction"),
        profiles={w: "flatfield_corr_matrix.png" for w in wavelengths},
    )

    with pytest.raises(ValueError, match="illumination"):
        illumination_correction(
            zarr_url=zarr_url,
            illumination_profiles=illum_profiles,
            overwrite_input=True,
            input_ROI_table="well_ROI_table",
        )


def test_darkfield_shape_mismatch(tmp_path: Path, testdata_path: Path) -> None:
    zarr_url, wavelengths = _make_tiny_zyx_zarr_with_roi(tmp_path)

    # Flatfield must match (10×10) so we get past that check
    tiny_ff = tmp_path / "tiny_flatfield.png"
    _save_tiny_flatfield(tiny_ff)
    illum_profiles = ProfileCorrectionModel(
        folder=str(tmp_path),
        profiles={w: "tiny_flatfield.png" for w in wavelengths},
    )
    # Darkfield PNG is 2160×2560 → mismatch with 10×10 image
    background = BackgroundCorrection(
        value=ProfileCorrectionModel(
            folder=str(testdata_path / "illumination_correction"),
            profiles={w: "darkfield_corr_matrix.png" for w in wavelengths},
            model="Profile",
        ),
    )

    with pytest.raises(ValueError, match=r"background \(darkfield\)"):
        illumination_correction(
            zarr_url=zarr_url,
            illumination_profiles=illum_profiles,
            background_correction=background,
            overwrite_input=True,
            input_ROI_table="well_ROI_table",
        )

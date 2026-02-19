"""
Integration tests for the full registration pipeline.

All fixtures are built synthetically with ngio — no Zenodo downloads required.

Pipeline under test:
  1. calculate_registration_image_based  — compute per-ROI pixel shifts
  2. find_registration_consensus         — derive consensus cropped region
  3. apply_registration_to_image         — write registered image to disk
"""

from pathlib import Path

import numpy as np
import pytest
from ngio import (
    ImageInWellPath,
    create_empty_ome_zarr,
    create_empty_plate,
    open_ome_zarr_container,
    open_ome_zarr_well,
)

from fractal_tasks_core.tasks._registration_utils_v2 import RegistrationMethod
from fractal_tasks_core.tasks.apply_registration_to_image import (
    apply_registration_to_image,
)
from fractal_tasks_core.tasks.calculate_registration_image_based import (
    calculate_registration_image_based,
)
from fractal_tasks_core.tasks.find_registration_consensus import (
    find_registration_consensus,
)
from fractal_tasks_core.tasks.io_models import (
    InitArgsRegistration,
    InitArgsRegistrationConsensus,
)

# ---------------------------------------------------------------------------
# Helpers / constants
# ---------------------------------------------------------------------------

# Image geometry
_SHAPE = (1, 1, 64, 64)  # czyx
_PIXELSIZE = 0.325  # µm/px at level 0
_LEVELS = 3  # level 2 = 16×16 px, pixel size = 0.325*4 = 1.3 µm/px
_WAVELENGTH = "A01_C01"

# Known shift applied to acquisition 1 (pixels at level 0)
_SHIFT_Y_PX = 4
_SHIFT_X_PX = 8
# At level 2 (factor 4) these become 1 px and 2 px, i.e. 1.3 µm and 2.6 µm.
_SHIFT_Y_UM = 1.3  # µm
_SHIFT_X_UM = 2.6  # µm


def _build_image(zarr_url: str, y_offset: int = 0, x_offset: int = 0) -> None:
    """Create a single-channel OME-Zarr image with a bright 10×10 block."""
    ome = create_empty_ome_zarr(
        zarr_url,
        shape=_SHAPE,
        xy_pixelsize=_PIXELSIZE,
        z_spacing=1.0,
        axes_names="czyx",
        levels=_LEVELS,
        channel_wavelengths=[_WAVELENGTH],
        overwrite=True,
    )
    img = ome.get_image()
    data = np.zeros(_SHAPE, dtype=np.uint16)
    y0, x0 = 20 + y_offset, 20 + x_offset
    data[0, 0, y0 : y0 + 10, x0 : x0 + 10] = 1000
    img.set_array(data)
    img.consolidate()
    fov = ome.build_image_roi_table("image")
    ome.add_table("FOV_ROI_table", fov, backend="experimental_json_v1")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def multiplex_plate_urls(tmp_path: Path) -> dict[str, str]:
    """
    Build a 2-acquisition plate in tmp_path.

    Acquisition 0 (reference): bright block at y=20:30, x=20:30.
    Acquisition 1 (to align):  same block shifted by +4 px (y) and +8 px (x).

    Returns a dict with keys 'zarr_url_0', 'zarr_url_1', 'well_url'.
    """
    plate_path = tmp_path / "multiplex.zarr"
    plate = create_empty_plate(
        store=plate_path,
        name="multiplex",
        images=[
            ImageInWellPath(row="A", column="01", path="0", acquisition_id=0),
            ImageInWellPath(row="A", column="01", path="1", acquisition_id=1),
        ],
        overwrite=True,
    )
    base_url = plate._group_handler.full_url  # ends with '/'
    zarr_url_0 = f"{base_url}A/01/0"
    zarr_url_1 = f"{base_url}A/01/1"
    well_url = f"{base_url}A/01"

    _build_image(zarr_url_0, y_offset=0, x_offset=0)
    _build_image(zarr_url_1, y_offset=_SHIFT_Y_PX, x_offset=_SHIFT_X_PX)

    return {"zarr_url_0": zarr_url_0, "zarr_url_1": zarr_url_1, "well_url": well_url}


# ---------------------------------------------------------------------------
# calculate_registration_image_based
# ---------------------------------------------------------------------------


def test_calculate_registration_stores_translations(multiplex_plate_urls):
    """After running calculate, acquisition 1's ROI table contains the detected shift."""
    zarr_url_0 = multiplex_plate_urls["zarr_url_0"]
    zarr_url_1 = multiplex_plate_urls["zarr_url_1"]

    calculate_registration_image_based(
        zarr_url=zarr_url_1,
        init_args=InitArgsRegistration(reference_zarr_url=zarr_url_0),
        wavelength_id=_WAVELENGTH,
        roi_table="FOV_ROI_table",
        level=2,
    )

    ome1 = open_ome_zarr_container(zarr_url_1)
    rois = ome1.get_generic_roi_table("FOV_ROI_table").rois()
    assert len(rois) == 1
    roi = rois[0]
    assert roi.model_extra is not None
    assert roi.model_extra["translation_y"] == pytest.approx(-_SHIFT_Y_UM, abs=0.1)
    assert roi.model_extra["translation_x"] == pytest.approx(-_SHIFT_X_UM, abs=0.1)
    assert roi.model_extra["translation_z"] == pytest.approx(0.0)

    # Reference acquisition should be unchanged (no translation fields)
    ome0 = open_ome_zarr_container(zarr_url_0)
    ref_rois = ome0.get_generic_roi_table("FOV_ROI_table").rois()
    assert not (ref_rois[0].model_extra or {}).get("translation_y")


def test_calculate_registration_chi2_shift_3d_raises(tmp_path: Path):
    """CHI2_SHIFT method is not supported for 3D images."""
    zarr_url = str(tmp_path / "img_3d.zarr")
    # 3D image: 1 channel, 4 z-slices, 32×32
    ome = create_empty_ome_zarr(
        zarr_url,
        shape=(1, 4, 32, 32),
        xy_pixelsize=0.325,
        z_spacing=1.0,
        axes_names="czyx",
        levels=3,
        channel_wavelengths=[_WAVELENGTH],
        overwrite=True,
    )
    fov = ome.build_image_roi_table("image")
    ome.add_table("FOV_ROI_table", fov, backend="experimental_json_v1")

    with pytest.raises(ValueError, match="CHI2_SHIFT"):
        calculate_registration_image_based(
            zarr_url=zarr_url,
            init_args=InitArgsRegistration(reference_zarr_url=zarr_url),
            wavelength_id=_WAVELENGTH,
            method=RegistrationMethod.CHI2_SHIFT,
            roi_table="FOV_ROI_table",
            level=2,
        )


def test_calculate_registration_time_series_raises(tmp_path: Path):
    """Time-series images (tczyx) are not supported."""
    zarr_url = str(tmp_path / "img_tczyx.zarr")
    ome = create_empty_ome_zarr(
        zarr_url,
        shape=(2, 1, 1, 32, 32),
        xy_pixelsize=0.325,
        z_spacing=1.0,
        axes_names="tczyx",
        levels=3,
        channel_wavelengths=[_WAVELENGTH],
        overwrite=True,
    )
    fov = ome.build_image_roi_table("image")
    ome.add_table("FOV_ROI_table", fov, backend="experimental_json_v1")

    with pytest.raises(ValueError, match="[Tt]ime"):
        calculate_registration_image_based(
            zarr_url=zarr_url,
            init_args=InitArgsRegistration(reference_zarr_url=zarr_url),
            wavelength_id=_WAVELENGTH,
            roi_table="FOV_ROI_table",
            level=2,
        )


# ---------------------------------------------------------------------------
# find_registration_consensus
# ---------------------------------------------------------------------------


def test_find_consensus_produces_correct_region(multiplex_plate_urls):
    """
    After calculate + consensus the registered ROI table should reflect the
    cropped overlap region.

    Reference ROI:   y=0.0,   x=0.0,   y_len≈19.5, x_len≈18.2
    Acquisition 1:   y=1.3,   x=2.6,   same size
    """
    zarr_url_0 = multiplex_plate_urls["zarr_url_0"]
    zarr_url_1 = multiplex_plate_urls["zarr_url_1"]

    calculate_registration_image_based(
        zarr_url=zarr_url_1,
        init_args=InitArgsRegistration(reference_zarr_url=zarr_url_0),
        wavelength_id=_WAVELENGTH,
        roi_table="FOV_ROI_table",
        level=2,
    )

    find_registration_consensus(
        zarr_url=zarr_url_0,
        init_args=InitArgsRegistrationConsensus(zarr_url_list=[zarr_url_0, zarr_url_1]),
        roi_table="FOV_ROI_table",
        new_roi_table="registered_FOV_ROI_table",
    )

    ome0 = open_ome_zarr_container(zarr_url_0)
    rois0 = ome0.get_roi_table("registered_FOV_ROI_table").rois()
    assert len(rois0) == 1
    ref_roi = rois0[0]
    assert ref_roi.y == pytest.approx(0.0, abs=0.01)
    assert ref_roi.x == pytest.approx(0.0, abs=0.01)
    assert ref_roi.y_length == pytest.approx(20.8 - _SHIFT_Y_UM, abs=0.1)
    assert ref_roi.x_length == pytest.approx(20.8 - _SHIFT_X_UM, abs=0.1)

    ome1 = open_ome_zarr_container(zarr_url_1)
    rois1 = ome1.get_roi_table("registered_FOV_ROI_table").rois()
    assert len(rois1) == 1
    acq_roi = rois1[0]
    assert acq_roi.y == pytest.approx(_SHIFT_Y_UM, abs=0.1)
    assert acq_roi.x == pytest.approx(_SHIFT_X_UM, abs=0.1)
    # Size must be identical to the reference
    assert acq_roi.y_length == pytest.approx(ref_roi.y_length, abs=0.01)
    assert acq_roi.x_length == pytest.approx(ref_roi.x_length, abs=0.01)


# ---------------------------------------------------------------------------
# apply_registration_to_image — full pipeline helpers
# ---------------------------------------------------------------------------


def _run_full_pipeline(zarr_url_0: str, zarr_url_1: str, overwrite_input: bool):
    """Run calculate → consensus → apply for the given pair."""
    calculate_registration_image_based(
        zarr_url=zarr_url_1,
        init_args=InitArgsRegistration(reference_zarr_url=zarr_url_0),
        wavelength_id=_WAVELENGTH,
        roi_table="FOV_ROI_table",
        level=2,
    )
    find_registration_consensus(
        zarr_url=zarr_url_0,
        init_args=InitArgsRegistrationConsensus(zarr_url_list=[zarr_url_0, zarr_url_1]),
        roi_table="FOV_ROI_table",
        new_roi_table="registered_FOV_ROI_table",
    )
    return apply_registration_to_image(
        zarr_url=zarr_url_1,
        registered_roi_table="registered_FOV_ROI_table",
        reference_acquisition=0,
        overwrite_input=overwrite_input,
    )


# ---------------------------------------------------------------------------
# apply_registration_to_image — overwrite_input=True
# ---------------------------------------------------------------------------


def test_full_pipeline_overwrite_input_true(multiplex_plate_urls):
    """
    Full pipeline with overwrite_input=True:
    - The original zarr_url_1 is replaced with the registered image in-place.
    - No '_registered' directory is left on disk.
    - The registered image is non-empty and carries the expected tables.
    """
    zarr_url_0 = multiplex_plate_urls["zarr_url_0"]
    zarr_url_1 = multiplex_plate_urls["zarr_url_1"]

    result = _run_full_pipeline(zarr_url_0, zarr_url_1, overwrite_input=True)

    # Return value
    assert result == {"image_list_updates": [{"zarr_url": zarr_url_1}]}

    # Original path still exists (now contains registered image)
    assert Path(zarr_url_1).exists()

    # No leftover '_registered' directory
    assert not Path(f"{zarr_url_1}_registered").exists()

    # Registered image is non-empty
    ome1 = open_ome_zarr_container(zarr_url_1)
    img1 = ome1.get_image()
    assert img1.get_array().any(), "Registered image is all zeros"

    # ROI tables are present
    tables = set(ome1.list_tables())
    assert "FOV_ROI_table" in tables
    assert "registered_FOV_ROI_table" in tables


# ---------------------------------------------------------------------------
# apply_registration_to_image — overwrite_input=False
# ---------------------------------------------------------------------------


def test_full_pipeline_overwrite_input_false(multiplex_plate_urls):
    """
    Full pipeline with overwrite_input=False:
    - A new '_registered' zarr is created alongside the original.
    - image_list_updates contains the new URL with origin.
    - The well metadata is updated to include the new image path.
    - The original zarr_url_1 is untouched.
    """
    zarr_url_0 = multiplex_plate_urls["zarr_url_0"]
    zarr_url_1 = multiplex_plate_urls["zarr_url_1"]
    well_url = multiplex_plate_urls["well_url"]
    new_zarr_url = f"{zarr_url_1}_registered"

    result = _run_full_pipeline(zarr_url_0, zarr_url_1, overwrite_input=False)

    # Return value
    updates = result["image_list_updates"]
    assert len(updates) == 1
    assert updates[0]["zarr_url"] == new_zarr_url
    assert updates[0]["origin"] == zarr_url_1

    # Both paths exist
    assert Path(zarr_url_1).exists()
    assert Path(new_zarr_url).exists()

    # Registered image is non-empty
    ome_new = open_ome_zarr_container(new_zarr_url)
    img_new = ome_new.get_image()
    assert img_new.get_array().any(), "Registered image is all zeros"

    # Well metadata updated
    well = open_ome_zarr_well(well_url)
    all_paths = well.paths()
    assert "1_registered" in all_paths

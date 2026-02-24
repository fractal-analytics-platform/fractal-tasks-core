"""
Integration test: full image-processing pipeline.

Pipeline under test:
  1. Create synthetic 2-acquisition plate (no ROI tables)
  2. import_ome_zarr        → adds image_ROI_table and grid_ROI_table
  3. illumination_correction → in-place correction using image_ROI_table
  4. Registration:
       a. image_based_registration_hcs_init
       b. calculate_registration_image_based
       c. find_registration_consensus
       d. apply_registration_to_image
  5. copy_ome_zarr_hcs_plate + projection  (MIP on acquisition 0)

All fixtures are built synthetically with ngio — no Zenodo downloads required.
The flatfield PNGs are also created synthetically in tmp_path.
"""

from pathlib import Path

import numpy as np
import pytest
from ngio import (
    ImageInWellPath,
    create_empty_ome_zarr,
    create_empty_plate,
    open_ome_zarr_container,
)
from skimage.io import imsave

from fractal_tasks_core._io_models import (
    InitArgsRegistration,
    InitArgsRegistrationConsensus,
    ProfileCorrectionModel,
)
from fractal_tasks_core.apply_registration_to_image import (
    apply_registration_to_image,
)
from fractal_tasks_core.calculate_registration_image_based import (
    calculate_registration_image_based,
)
from fractal_tasks_core.copy_ome_zarr_hcs_plate import copy_ome_zarr_hcs_plate
from fractal_tasks_core.find_registration_consensus import (
    find_registration_consensus,
)
from fractal_tasks_core.illumination_correction import illumination_correction
from fractal_tasks_core.image_based_registration_hcs_init import (
    image_based_registration_hcs_init,
)
from fractal_tasks_core.import_ome_zarr import import_ome_zarr
from fractal_tasks_core.projection import projection

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CHANNELS = ["A01_C01", "A01_C02"]
_SHAPE = (2, 4, 64, 64)  # czyx: 2 channels, 4 z-slices, 64×64 px
_PIXELSIZE = 0.325  # µm/px at level 0
_Z_SPACING = 1.0  # µm/z-slice
_LEVELS = 3  # pyramid levels; level 2 = 4× downsampled → 16×16 px

# Known shift applied to acquisition 1 relative to acquisition 0 (level-0 px)
_SHIFT_Y_PX = 4
_SHIFT_X_PX = 8
# At level 2 (4× xy downsampling):
#   y:  4 px → 1 px × (0.325 × 4) µm/px = 1.3 µm
#   x:  8 px → 2 px × (0.325 × 4) µm/px = 2.6 µm
_SHIFT_Y_UM = _SHIFT_Y_PX * _PIXELSIZE  # 1.3 µm
_SHIFT_X_UM = _SHIFT_X_PX * _PIXELSIZE  # 2.6 µm

# ROI table names — reused from import_ome_zarr output
_ROI_TABLE = "image_ROI_table"
_REGISTERED_ROI_TABLE = "registered_image_ROI_table"


# ---------------------------------------------------------------------------
# Plate builder
# ---------------------------------------------------------------------------


def _build_plate(tmp_path: Path) -> tuple[str, str, str]:
    """
    Create a 2-acquisition, 1-well plate with a bright 10×10 block.

    Acquisition 0 (reference): block at y=[20:30], x=[20:30].
    Acquisition 1 (to align):  same block shifted by (_SHIFT_Y_PX, _SHIFT_X_PX).

    No ROI tables are written here — import_ome_zarr will add them.

    Returns: (zarr_url_0, zarr_url_1, well_url).
    """
    plate_path = tmp_path / "pipeline_plate.zarr"
    create_empty_plate(
        store=plate_path,
        name="pipeline_plate",
        images=[
            ImageInWellPath(row="A", column="01", path="0", acquisition_id=0),
            ImageInWellPath(row="A", column="01", path="1", acquisition_id=1),
        ],
        overwrite=True,
    )
    base_url = plate_path.as_posix()
    zarr_url_0 = f"{base_url}/A/01/0"
    zarr_url_1 = f"{base_url}/A/01/1"
    well_url = f"{base_url}/A/01"

    for zarr_url, y_off, x_off in [
        (zarr_url_0, 0, 0),
        (zarr_url_1, _SHIFT_Y_PX, _SHIFT_X_PX),
    ]:
        ome = create_empty_ome_zarr(
            zarr_url,
            shape=_SHAPE,
            pixelsize=_PIXELSIZE,
            z_spacing=_Z_SPACING,
            axes_names="czyx",
            levels=_LEVELS,
            channel_wavelengths=_CHANNELS,
            overwrite=True,
        )
        data = np.zeros(_SHAPE, dtype=np.uint16)
        data[:, :, 20 + y_off : 30 + y_off, 20 + x_off : 30 + x_off] = 1_000
        img = ome.get_image()
        img.set_array(data)
        img.consolidate()

    return zarr_url_0, zarr_url_1, well_url


# ---------------------------------------------------------------------------
# Flatfield PNG builder
# ---------------------------------------------------------------------------


def _create_flatfields(tmp_path: Path) -> tuple[str, dict[str, str]]:
    """
    Save 64×64 flatfield PNG files (one per channel) to tmp_path/flatfields/.

    The profile is a Gaussian centred at (32, 32) with values in [1 000, 5 000].
    After ``correction_matrix / max(correction_matrix)`` the profile is in
    (0.2, 1.0] — no zeros, so the zero-check in illumination_correction passes
    and the correction is non-trivial (pixel values change from their input).

    Returns: (folder_path_str, {wavelength_id: filename}).
    """
    illum_dir = tmp_path / "flatfields"
    illum_dir.mkdir(exist_ok=True)

    yy, xx = np.mgrid[0:64, 0:64]
    profile = (
        1_000 + 4_000 * np.exp(-((yy - 32) ** 2 + (xx - 32) ** 2) / (2 * 20.0**2))
    ).astype(np.uint16)

    profiles: dict[str, str] = {}
    for channel in _CHANNELS:
        fname = f"flatfield_{channel}.png"
        imsave(str(illum_dir / fname), profile)
        profiles[channel] = fname

    return str(illum_dir), profiles


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


def test_full_pipeline(tmp_path: Path) -> None:
    """
    End-to-end pipeline: import → illumination correction → registration → projection.
    """

    # ------------------------------------------------------------------
    # 1. Build synthetic plate (no ROI tables yet)
    # ------------------------------------------------------------------
    zarr_url_0, zarr_url_1, _ = _build_plate(tmp_path)
    illum_dir, profiles = _create_flatfields(tmp_path)

    # ------------------------------------------------------------------
    # 2. import_ome_zarr — creates image_ROI_table and grid_ROI_table
    # ------------------------------------------------------------------
    import_result = import_ome_zarr(
        zarr_dir=str(tmp_path),
        zarr_name="pipeline_plate.zarr",
        grid_y_shape=1,
        grid_x_shape=1,
        update_omero_metadata=False,
    )
    updates = import_result["image_list_updates"]
    assert len(updates) == 2, f"Expected 2 image updates, got {len(updates)}"
    for update in updates:
        tables = open_ome_zarr_container(update["zarr_url"]).list_tables()
        assert "image_ROI_table" in tables, "import_ome_zarr must add image_ROI_table"
        assert "grid_ROI_table" in tables, "import_ome_zarr must add grid_ROI_table"

    # ------------------------------------------------------------------
    # 3. illumination_correction (overwrite_input=True, image_ROI_table)
    # ------------------------------------------------------------------
    illumination_profiles = ProfileCorrectionModel(folder=illum_dir, profiles=profiles)
    for zarr_url in [zarr_url_0, zarr_url_1]:
        illumination_correction(
            zarr_url=zarr_url,
            illumination_profiles=illumination_profiles,
            overwrite_input=True,
            input_ROI_table=_ROI_TABLE,
        )

    # Both images should still exist and be non-empty after in-place correction
    for zarr_url in [zarr_url_0, zarr_url_1]:
        assert open_ome_zarr_container(zarr_url).get_image().get_array().any(), (
            f"Image at {zarr_url} is all zeros after illumination correction"
        )

    # ------------------------------------------------------------------
    # 4a. Registration — init: build parallelization list
    # ------------------------------------------------------------------
    init_result = image_based_registration_hcs_init(
        zarr_urls=[zarr_url_0, zarr_url_1],
        zarr_dir="/unused",
        reference_acquisition=0,
    )
    para_list = init_result["parallelization_list"]
    assert len(para_list) == 1, "Expected one non-reference acquisition"
    assert para_list[0]["zarr_url"] == zarr_url_1
    assert para_list[0]["init_args"]["reference_zarr_url"] == zarr_url_0

    # ------------------------------------------------------------------
    # 4b. Registration — calculate per-ROI shifts for acquisition 1
    # ------------------------------------------------------------------
    calculate_registration_image_based(
        zarr_url=zarr_url_1,
        init_args=InitArgsRegistration(reference_zarr_url=zarr_url_0),
        wavelength_id=_CHANNELS[0],
        roi_table=_ROI_TABLE,
        level=2,
    )

    ome1 = open_ome_zarr_container(zarr_url_1)
    rois = ome1.get_generic_roi_table(_ROI_TABLE).rois()
    assert len(rois) == 1
    roi = rois[0]
    assert roi.model_extra is not None
    assert roi.model_extra["translation_y"] == pytest.approx(-_SHIFT_Y_UM, abs=0.2)
    assert roi.model_extra["translation_x"] == pytest.approx(-_SHIFT_X_UM, abs=0.2)
    assert roi.model_extra["translation_z"] == pytest.approx(0.0, abs=0.5)

    # ------------------------------------------------------------------
    # 4c. Registration — consensus: derive registered overlap region
    # ------------------------------------------------------------------
    find_registration_consensus(
        zarr_url=zarr_url_0,
        init_args=InitArgsRegistrationConsensus(zarr_url_list=[zarr_url_0, zarr_url_1]),
        roi_table=_ROI_TABLE,
        new_roi_table=_REGISTERED_ROI_TABLE,
    )

    ome0 = open_ome_zarr_container(zarr_url_0)
    assert _REGISTERED_ROI_TABLE in ome0.list_tables(), (
        "find_registration_consensus must write registered_image_ROI_table to acq-0"
    )

    # ------------------------------------------------------------------
    # 4d. Registration — apply: write registered acquisition 1 in-place
    # ------------------------------------------------------------------
    apply_result = apply_registration_to_image(
        zarr_url=zarr_url_1,
        registered_roi_table=_REGISTERED_ROI_TABLE,
        reference_acquisition=0,
        overwrite_input=True,
    )
    assert apply_result == {"image_list_updates": [{"zarr_url": zarr_url_1}]}
    assert Path(zarr_url_1).exists()
    assert open_ome_zarr_container(zarr_url_1).get_image().get_array().any(), (
        "Registered image (acq 1) must be non-empty"
    )

    # ------------------------------------------------------------------
    # 5. Projection — MIP of acquisition 0 (3D czyx → z=1)
    # ------------------------------------------------------------------
    proj_dir = str(tmp_path / "projections")
    para_proj = copy_ome_zarr_hcs_plate(
        zarr_urls=[zarr_url_0],
        zarr_dir=proj_dir,
        overwrite=True,
        re_initialize_plate=True,
    )
    assert len(para_proj["parallelization_list"]) == 1

    proj_item = para_proj["parallelization_list"][0]
    proj_result = projection(**proj_item)

    proj_zarr_url = proj_result["image_list_updates"][0]["zarr_url"]
    assert Path(proj_zarr_url).exists(), "Projected zarr must exist on disk"

    proj_ome = open_ome_zarr_container(proj_zarr_url)
    proj_img = proj_ome.get_image()
    assert proj_img.dimensions.get("z", default=None) == 1, (
        "Projected image must have z=1"
    )
    assert proj_img.get_array().any(), "Projected image must be non-empty"

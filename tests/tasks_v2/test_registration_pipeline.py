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
    Roi,
    create_empty_ome_zarr,
    create_empty_plate,
    open_ome_zarr_container,
    open_ome_zarr_well,
)
from ngio.tables import FeatureTable, RoiTable
from pandas import DataFrame

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
from fractal_tasks_core.tasks.image_based_registration_hcs_init import (
    image_based_registration_hcs_init,
)
from fractal_tasks_core.tasks.init_group_by_well_for_multiplexing import (
    init_group_by_well_for_multiplexing,
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
# z shift: ngio only downsamples xy, not z.  z_spacing = 1.0 µm/px at every
# pyramid level, so 2 px × 1.0 µm/px = 2.0 µm.
_SHIFT_Z_PX = 2
_SHIFT_Z_UM = 2.0  # µm
# Table backend
_TABLE_BACKEND = "anndata"


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
    ome.add_table("FOV_ROI_table", fov, backend=_TABLE_BACKEND)


def _build_image_for_axes(
    zarr_url: str,
    axes: str,
    shape: tuple[int, ...],
    y_offset: int = 0,
    x_offset: int = 0,
) -> None:
    """Create an OME-Zarr image for any axes configuration with a bright 10×10 block.

    ngio supports channel metadata even for squeezed/implicit channels, so
    _WAVELENGTH is always provided.  z_spacing is added only when 'z' is present.

    The z axis is left as slice(None) (signal across all z slices).  This keeps
    the signal density high enough for phase_cross_correlation and avoids the
    ngio behaviour where channel_selection picks the z=0 slice for images
    without an explicit 'c' axis.
    """
    kwargs: dict = dict(
        shape=shape,
        xy_pixelsize=_PIXELSIZE,
        axes_names=axes,
        levels=_LEVELS,
        channel_wavelengths=[_WAVELENGTH],
        overwrite=True,
    )
    if "z" in axes:
        kwargs["z_spacing"] = 1.0
    ome = create_empty_ome_zarr(zarr_url, **kwargs)
    img = ome.get_image()
    data = np.zeros(shape, dtype=np.uint16)
    y_idx = axes.index("y")
    x_idx = axes.index("x")
    slices: list = [slice(None)] * len(shape)
    slices[y_idx] = slice(20 + y_offset, 30 + y_offset)
    slices[x_idx] = slice(20 + x_offset, 30 + x_offset)
    data[tuple(slices)] = 1000
    img.set_array(data)
    img.consolidate()
    fov = ome.build_image_roi_table("image")
    ome.add_table("FOV_ROI_table", fov, backend=_TABLE_BACKEND)


def _build_multi_fov_image(zarr_url: str, y_offset: int = 0, x_offset: int = 0) -> None:
    """Create a czyx image with two FOVs (top/bottom halves) and one bright block each.

    FOV_1 covers rows 0–32 (y ∈ [0, 10.4) µm).
    FOV_2 covers rows 32–64 (y ∈ [10.4, 20.8) µm).
    The bright 10×10 block is placed at the same relative position (row 10, col 10)
    within each half, so both FOVs experience the same (y_offset, x_offset) shift.
    """
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
    data = np.zeros(_SHAPE, dtype=np.uint16)
    y0, x0 = 10 + y_offset, 10 + x_offset
    # FOV_1: block in top half (rows 0–32)
    data[0, 0, y0 : y0 + 10, x0 : x0 + 10] = 1000
    # FOV_2: block in bottom half (rows 32–64); same relative position within the half
    data[0, 0, 32 + y0 : 32 + y0 + 10, x0 : x0 + 10] = 1000
    ome.get_image().set_array(data)
    ome.get_image().consolidate()

    half_um = (_SHAPE[-2] // 2) * _PIXELSIZE  # 10.4 µm
    full_um = _SHAPE[-1] * _PIXELSIZE  # 20.8 µm
    fov1 = Roi(name="FOV_1", y=0.0, x=0.0, y_length=half_um, x_length=full_um)
    fov2 = Roi(name="FOV_2", y=half_um, x=0.0, y_length=half_um, x_length=full_um)
    ome.add_table("FOV_ROI_table", RoiTable(rois=[fov1, fov2]), backend=_TABLE_BACKEND)


def _build_multiplex_plate(
    tmp_path: Path, axes: str, shape: tuple[int, ...]
) -> dict[str, str]:
    """Build a 2-acquisition plate for any axes/shape with a known shift.

    Acquisition 0 (reference): bright block at y=20:30, x=20:30.
    Acquisition 1 (to align):  block shifted by _SHIFT_Y_PX / _SHIFT_X_PX.
    """
    plate_path = tmp_path / f"multiplex_{axes}.zarr"
    plate = create_empty_plate(
        store=plate_path,
        name=f"multiplex_{axes}",
        images=[
            ImageInWellPath(row="A", column="01", path="0", acquisition_id=0),
            ImageInWellPath(row="A", column="01", path="1", acquisition_id=1),
        ],
        overwrite=True,
    )
    base_url = plate._group_handler.full_url
    zarr_url_0 = f"{base_url}A/01/0"
    zarr_url_1 = f"{base_url}A/01/1"
    well_url = f"{base_url}A/01"
    _build_image_for_axes(zarr_url_0, axes, shape)
    _build_image_for_axes(
        zarr_url_1, axes, shape, y_offset=_SHIFT_Y_PX, x_offset=_SHIFT_X_PX
    )
    return {"zarr_url_0": zarr_url_0, "zarr_url_1": zarr_url_1, "well_url": well_url}


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
# image_based_registration_hcs_init
# ---------------------------------------------------------------------------


def test_init_produces_correct_parallelization_list(multiplex_plate_urls):
    """Init returns one entry: acq-1 paired with acq-0 as reference."""
    zarr_url_0 = multiplex_plate_urls["zarr_url_0"]
    zarr_url_1 = multiplex_plate_urls["zarr_url_1"]

    result = image_based_registration_hcs_init(
        zarr_urls=[zarr_url_0, zarr_url_1],
        zarr_dir="/unused",
        reference_acquisition=0,
    )

    plist = result["parallelization_list"]
    assert len(plist) == 1
    assert plist[0]["zarr_url"] == zarr_url_1
    assert plist[0]["init_args"]["reference_zarr_url"] == zarr_url_0


def test_init_excludes_reference_from_list(multiplex_plate_urls):
    """The reference acquisition must not appear in the parallelization list."""
    zarr_url_0 = multiplex_plate_urls["zarr_url_0"]
    zarr_url_1 = multiplex_plate_urls["zarr_url_1"]

    result = image_based_registration_hcs_init(
        zarr_urls=[zarr_url_0, zarr_url_1],
        zarr_dir="/unused",
        reference_acquisition=0,
    )

    urls_in_list = [entry["zarr_url"] for entry in result["parallelization_list"]]
    assert zarr_url_0 not in urls_in_list


def test_init_missing_reference_raises(tmp_path: Path):
    """ValueError when reference_acquisition is absent from the plate metadata."""
    plate_path = tmp_path / "missing_ref.zarr"
    plate = create_empty_plate(
        store=plate_path,
        name="missing_ref",
        images=[
            ImageInWellPath(row="A", column="01", path="1", acquisition_id=1),
        ],
        overwrite=True,
    )
    base_url = plate._group_handler.full_url
    zarr_url_1 = f"{base_url}A/01/1"
    _build_image(zarr_url_1)

    with pytest.raises(ValueError, match="[Nn]o reference acquisition"):
        image_based_registration_hcs_init(
            zarr_urls=[zarr_url_1],
            zarr_dir="/unused",
            reference_acquisition=0,
        )


# ---------------------------------------------------------------------------
# init_group_by_well_for_multiplexing
# ---------------------------------------------------------------------------


def test_group_by_well_produces_correct_parallelization_list(multiplex_plate_urls):
    """Group-by-well init returns one entry per well with all acquisition URLs."""
    zarr_url_0 = multiplex_plate_urls["zarr_url_0"]
    zarr_url_1 = multiplex_plate_urls["zarr_url_1"]

    result = init_group_by_well_for_multiplexing(
        zarr_urls=[zarr_url_0, zarr_url_1],
        zarr_dir="/unused",
        reference_acquisition=0,
    )

    plist = result["parallelization_list"]
    assert len(plist) == 1
    assert plist[0]["zarr_url"] == zarr_url_0
    assert set(plist[0]["init_args"]["zarr_url_list"]) == {zarr_url_0, zarr_url_1}


def test_group_by_well_missing_reference_raises(tmp_path: Path):
    """ValueError when reference_acquisition is absent from the well metadata."""
    plate_path = tmp_path / "missing_ref2.zarr"
    plate = create_empty_plate(
        store=plate_path,
        name="missing_ref2",
        images=[
            ImageInWellPath(row="A", column="01", path="1", acquisition_id=1),
        ],
        overwrite=True,
    )
    base_url = plate._group_handler.full_url
    zarr_url_1 = f"{base_url}A/01/1"
    _build_image(zarr_url_1)

    with pytest.raises(ValueError):
        init_group_by_well_for_multiplexing(
            zarr_urls=[zarr_url_1],
            zarr_dir="/unused",
            reference_acquisition=0,
        )


# ---------------------------------------------------------------------------
# calculate_registration_image_based
# ---------------------------------------------------------------------------


def test_calculate_registration_stores_translations(multiplex_plate_urls):
    """After running calculate, acquisition ROI table contains the detected shift."""
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


def test_calculate_registration_detects_z_shift(tmp_path: Path):
    """calculate_registration_image_based detects a pure z-shift in czyx images.

    Two design choices keep this test reliable:
    - 8 z-slices of signal (out of 16): at level 2 (xy/4, z unchanged because
      ngio only downsamples xy) the 10×10 block shrinks to ~3×3 px.  Using 8
      z-slices gives ~72 signal voxels out of 4 096 (1.76 %), which keeps
      np.quantile(img, 0.99) above zero so rescale_intensity does not clip.
    - czyx axes (not zyx): for zyx images get_roi with channel_selection picks
      the z=0 slice; czyx ensures the full (z, y, x) volume is returned.
    """
    shape = (1, 16, 64, 64)  # czyx
    _z_block = 8  # 8 z-slices of signal → ~1.76 % density at level 2
    _z_start = 4  # start well away from boundaries

    plate_path = tmp_path / "multiplex_zshift.zarr"
    plate = create_empty_plate(
        store=plate_path,
        name="multiplex_zshift",
        images=[
            ImageInWellPath(row="A", column="01", path="0", acquisition_id=0),
            ImageInWellPath(row="A", column="01", path="1", acquisition_id=1),
        ],
        overwrite=True,
    )
    base_url = plate._group_handler.full_url
    zarr_url_0 = f"{base_url}A/01/0"
    zarr_url_1 = f"{base_url}A/01/1"

    for zarr_url, z_start in [
        (zarr_url_0, _z_start),
        (zarr_url_1, _z_start + _SHIFT_Z_PX),
    ]:
        ome = create_empty_ome_zarr(
            zarr_url,
            shape=shape,
            xy_pixelsize=_PIXELSIZE,
            z_spacing=1.0,
            axes_names="czyx",
            levels=_LEVELS,
            channel_wavelengths=[_WAVELENGTH],
            overwrite=True,
        )
        data = np.zeros(shape, dtype=np.uint16)
        data[0, z_start : z_start + _z_block, 20:30, 20:30] = 1000
        ome.get_image().set_array(data)
        ome.get_image().consolidate()
        fov = ome.build_image_roi_table("image")
        ome.add_table("FOV_ROI_table", fov, backend=_TABLE_BACKEND)

    calculate_registration_image_based(
        zarr_url=zarr_url_1,
        init_args=InitArgsRegistration(reference_zarr_url=zarr_url_0),
        wavelength_id=_WAVELENGTH,
        roi_table="FOV_ROI_table",
        level=2,  # ngio only downsamples xy; z shape is unchanged at every level
    )

    ome1 = open_ome_zarr_container(zarr_url_1)
    roi = ome1.get_generic_roi_table("FOV_ROI_table").rois()[0]
    assert roi.model_extra is not None
    assert roi.model_extra["translation_z"] == pytest.approx(-_SHIFT_Z_UM, abs=0.5)
    assert roi.model_extra["translation_y"] == pytest.approx(0.0, abs=0.5)
    assert roi.model_extra["translation_x"] == pytest.approx(0.0, abs=0.5)


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
    ome.add_table("FOV_ROI_table", fov, backend=_TABLE_BACKEND)

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
    ome.add_table("FOV_ROI_table", fov, backend=_TABLE_BACKEND)

    with pytest.raises(ValueError, match="[Tt]ime"):
        calculate_registration_image_based(
            zarr_url=zarr_url,
            init_args=InitArgsRegistration(reference_zarr_url=zarr_url),
            wavelength_id=_WAVELENGTH,
            roi_table="FOV_ROI_table",
            level=2,
        )


# ---------------------------------------------------------------------------
# calculate_registration_image_based — multi-FOV
# ---------------------------------------------------------------------------


def test_calculate_registration_multi_fov(tmp_path: Path):
    """calculate processes N>1 ROIs independently and writes one shift per FOV.

    Both FOV_1 (top half) and FOV_2 (bottom half) have an identical bright block
    at the same relative position within their half, shifted by the same known
    amount in acquisition 1. After calculate, both ROIs must carry the correct
    independent translations.
    """
    plate_path = tmp_path / "multi_fov.zarr"
    plate = create_empty_plate(
        store=plate_path,
        name="multi_fov",
        images=[
            ImageInWellPath(row="A", column="01", path="0", acquisition_id=0),
            ImageInWellPath(row="A", column="01", path="1", acquisition_id=1),
        ],
        overwrite=True,
    )
    base_url = plate._group_handler.full_url
    zarr_url_0 = f"{base_url}A/01/0"
    zarr_url_1 = f"{base_url}A/01/1"

    _build_multi_fov_image(zarr_url_0)
    _build_multi_fov_image(zarr_url_1, y_offset=_SHIFT_Y_PX, x_offset=_SHIFT_X_PX)

    calculate_registration_image_based(
        zarr_url=zarr_url_1,
        init_args=InitArgsRegistration(reference_zarr_url=zarr_url_0),
        wavelength_id=_WAVELENGTH,
        roi_table="FOV_ROI_table",
        level=2,
    )

    ome1 = open_ome_zarr_container(zarr_url_1)
    rois = {r.name: r for r in ome1.get_generic_roi_table("FOV_ROI_table").rois()}
    assert len(rois) == 2
    for name in ("FOV_1", "FOV_2"):
        roi = rois[name]
        assert roi.model_extra is not None, f"{name} has no model_extra"
        assert roi.model_extra["translation_y"] == pytest.approx(
            -_SHIFT_Y_UM, abs=0.2
        ), name
        assert roi.model_extra["translation_x"] == pytest.approx(
            -_SHIFT_X_UM, abs=0.2
        ), name
        assert roi.model_extra["translation_z"] == pytest.approx(0.0, abs=0.1), name


# ---------------------------------------------------------------------------
# Full pipeline — multi-FOV (N>1 ROIs through consensus and apply)
# ---------------------------------------------------------------------------


def test_full_pipeline_multi_fov(tmp_path: Path):
    """Full pipeline with 2 FOVs covers consensus and apply with N>1 ROIs.

    Exercises:
    - _group_roi_by_name with 2 named ROIs across acquisitions
    - _apply_consensus_to_roi_table iterating 2 rows
    - apply_registration_to_image writing both FOV regions

    After apply, both acquisitions must have pixel-identical data in each
    registered FOV region (using the reference ROI for both).
    """
    plate_path = tmp_path / "multi_fov_pipeline.zarr"
    plate = create_empty_plate(
        store=plate_path,
        name="multi_fov_pipeline",
        images=[
            ImageInWellPath(row="A", column="01", path="0", acquisition_id=0),
            ImageInWellPath(row="A", column="01", path="1", acquisition_id=1),
        ],
        overwrite=True,
    )
    base_url = plate._group_handler.full_url
    zarr_url_0 = f"{base_url}A/01/0"
    zarr_url_1 = f"{base_url}A/01/1"

    _build_multi_fov_image(zarr_url_0)
    _build_multi_fov_image(zarr_url_1, y_offset=_SHIFT_Y_PX, x_offset=_SHIFT_X_PX)

    result = _run_full_pipeline(zarr_url_0, zarr_url_1, overwrite_input=True)

    assert result == {"image_list_updates": [{"zarr_url": zarr_url_1}]}
    assert Path(zarr_url_1).exists()

    ome0 = open_ome_zarr_container(zarr_url_0)
    ome1 = open_ome_zarr_container(zarr_url_1)
    img1 = ome1.get_image()

    assert img1.get_array().any(), "Registered multi-FOV image is all zeros"

    # The registered_FOV_ROI_table on acq0 has 2 entries (FOV_1, FOV_2).
    # After apply, acq1's data is in reference coordinates, so using the same
    # ref_roi extracts the same physical patch from both acquisitions.
    registered_table = ome0.get_roi_table("registered_FOV_ROI_table")
    ref_rois = {r.name: r for r in registered_table.rois()}
    assert len(ref_rois) == 2, "Expected 2 registered ROIs (FOV_1 and FOV_2)"
    for name, ref_roi in ref_rois.items():
        patch0 = ome0.get_image().get_roi(ref_roi)
        patch1 = img1.get_roi(ref_roi)
        np.testing.assert_array_equal(
            patch0, patch1, err_msg=f"Pixel mismatch in {name}"
        )


# ---------------------------------------------------------------------------
# calculate_registration_image_based — chi2_shift 2D success
# ---------------------------------------------------------------------------


def test_calculate_registration_chi2_shift_2d(multiplex_plate_urls):
    """CHI2_SHIFT detects the correct shift on a (effectively) 2D image.

    The czyx plate (1, 1, 64, 64) squeezes to (y, x) inside chi2_shift_out
    via np.squeeze, satisfying the 2D-only requirement.
    """
    zarr_url_0 = multiplex_plate_urls["zarr_url_0"]
    zarr_url_1 = multiplex_plate_urls["zarr_url_1"]

    calculate_registration_image_based(
        zarr_url=zarr_url_1,
        init_args=InitArgsRegistration(reference_zarr_url=zarr_url_0),
        wavelength_id=_WAVELENGTH,
        method=RegistrationMethod.CHI2_SHIFT,
        roi_table="FOV_ROI_table",
        level=2,
    )

    ome1 = open_ome_zarr_container(zarr_url_1)
    roi = ome1.get_generic_roi_table("FOV_ROI_table").rois()[0]
    assert roi.model_extra is not None
    assert roi.model_extra["translation_y"] == pytest.approx(-_SHIFT_Y_UM, abs=0.2)
    assert roi.model_extra["translation_x"] == pytest.approx(-_SHIFT_X_UM, abs=0.2)


# ---------------------------------------------------------------------------
# calculate_registration_image_based — shape mismatch
# ---------------------------------------------------------------------------


def test_calculate_registration_shape_mismatch_raises(tmp_path: Path):
    """NotImplementedError when the two acquisitions have different pixel shapes.

    acq0: 64×64 px at 0.325 µm/px (20.8 µm FOV).
    acq1: 48×48 px at 0.325 µm/px (15.6 µm FOV).
    The full-image ROI from acq0 covers 20.8 µm; at level 2 that extracts a
    16×16 array from acq0 but only a 12×12 array from acq1 → shape mismatch.
    """
    plate_path = tmp_path / "shape_mismatch.zarr"
    plate = create_empty_plate(
        store=plate_path,
        name="shape_mismatch",
        images=[
            ImageInWellPath(row="A", column="01", path="0", acquisition_id=0),
            ImageInWellPath(row="A", column="01", path="1", acquisition_id=1),
        ],
        overwrite=True,
    )
    base_url = plate._group_handler.full_url
    zarr_url_0 = f"{base_url}A/01/0"
    zarr_url_1 = f"{base_url}A/01/1"

    for zarr_url, shape in [(zarr_url_0, (1, 1, 64, 64)), (zarr_url_1, (1, 1, 48, 48))]:
        ome = create_empty_ome_zarr(
            zarr_url,
            shape=shape,
            xy_pixelsize=_PIXELSIZE,
            z_spacing=1.0,
            axes_names="czyx",
            levels=_LEVELS,
            channel_wavelengths=[_WAVELENGTH],
            overwrite=True,
        )
        data = np.zeros(shape, dtype=np.uint16)
        data[0, 0, 10:20, 10:20] = 1000
        ome.get_image().set_array(data)
        ome.get_image().consolidate()
        fov = ome.build_image_roi_table("image")
        ome.add_table("FOV_ROI_table", fov, backend=_TABLE_BACKEND)

    with pytest.raises(NotImplementedError, match="[Ss]hape"):
        calculate_registration_image_based(
            zarr_url=zarr_url_1,
            init_args=InitArgsRegistration(reference_zarr_url=zarr_url_0),
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

    # After consensus: get_roi on both acquisitions with their respective registered
    # ROIs should return pixel-identical arrays (same physical overlap region,
    # but addressed by different pixel coordinates in each acquisition).
    patch0 = ome0.get_image().get_roi(ref_roi)
    patch1 = ome1.get_image().get_roi(acq_roi)
    np.testing.assert_array_equal(patch0, patch1)


def test_consensus_mismatched_roi_names_raises(tmp_path: Path):
    """find_registration_consensus raises when acquisitions have different ROI names."""
    plate_path = tmp_path / "mismatch.zarr"
    plate = create_empty_plate(
        store=plate_path,
        name="mismatch",
        images=[
            ImageInWellPath(row="A", column="01", path="0", acquisition_id=0),
            ImageInWellPath(row="A", column="01", path="1", acquisition_id=1),
        ],
        overwrite=True,
    )
    base_url = plate._group_handler.full_url
    zarr_url_0 = f"{base_url}A/01/0"
    zarr_url_1 = f"{base_url}A/01/1"

    # acq0 gets ROI named "FOV_A", acq1 gets ROI named "FOV_B"
    for zarr_url, roi_name in [(zarr_url_0, "FOV_A"), (zarr_url_1, "FOV_B")]:
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
        fov = ome.build_image_roi_table(roi_name)
        ome.add_table("FOV_ROI_table", fov, backend=_TABLE_BACKEND)

    with pytest.raises(ValueError, match="[Rr]OI"):
        find_registration_consensus(
            zarr_url=zarr_url_0,
            init_args=InitArgsRegistrationConsensus(
                zarr_url_list=[zarr_url_0, zarr_url_1]
            ),
            roi_table="FOV_ROI_table",
        )


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

    # Add some non-ROI table
    ome1 = open_ome_zarr_container(zarr_url_1)
    # example feature table with dummy data
    features = DataFrame({"label": [1], "feature1": [0], "feature2": [1]})
    features_table = FeatureTable(features)
    ome1.add_table("example_feature_table", features_table, backend=_TABLE_BACKEND)

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

    # Feature table (non-ROI) is copied from the acquisition unchanged.
    # ngio stores 'label' as the DataFrame index, not as a regular column.
    assert "example_feature_table" in tables
    ft = ome1.get_table("example_feature_table")
    assert len(ft.dataframe) == 1
    assert list(ft.dataframe.columns) == ["feature1", "feature2"]

    # After apply: both acquisitions must have pixel-identical data in the
    # registered overlap region.  apply writes acq1 into the reference coordinate
    # frame, so the same ref_roi extracts the same physical patch from both.
    ome0 = open_ome_zarr_container(zarr_url_0)
    ref_roi = ome0.get_roi_table("registered_FOV_ROI_table").rois()[0]
    patch0 = ome0.get_image().get_roi(ref_roi)
    patch1 = img1.get_roi(ref_roi)
    np.testing.assert_array_equal(patch0, patch1)


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


# ---------------------------------------------------------------------------
# apply_registration_to_image — reference acquisition
# ---------------------------------------------------------------------------


def test_apply_on_reference_acquisition(multiplex_plate_urls):
    """apply called on the reference acquisition itself completes without error.

    When zarr_url == reference_zarr_url the warning about non-ROI tables is
    suppressed (line 226 of apply). This test covers that branch.
    """
    zarr_url_0 = multiplex_plate_urls["zarr_url_0"]
    zarr_url_1 = multiplex_plate_urls["zarr_url_1"]

    # Run calculate + consensus so the registered ROI table exists on both
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

    # Apply on the reference itself (zarr_url == reference_zarr_url)
    result = apply_registration_to_image(
        zarr_url=zarr_url_0,
        registered_roi_table="registered_FOV_ROI_table",
        reference_acquisition=0,
        overwrite_input=True,
    )

    assert result == {"image_list_updates": [{"zarr_url": zarr_url_0}]}
    assert Path(zarr_url_0).exists()


# ---------------------------------------------------------------------------
# apply_registration_to_image — label images
# ---------------------------------------------------------------------------


def test_full_pipeline_with_labels(multiplex_plate_urls):
    """apply copies and registers label images when they exist on the acquisition.

    Covers the `label_list` branch (apply lines 204–210) that is always False
    in other tests.
    """
    zarr_url_0 = multiplex_plate_urls["zarr_url_0"]
    zarr_url_1 = multiplex_plate_urls["zarr_url_1"]

    # Add a label to acquisition 1 before running the pipeline.
    # derive_label strips the channel axis: shape becomes (z, y, x) = (1, 64, 64).
    ome1_pre = open_ome_zarr_container(zarr_url_1)
    label = ome1_pre.derive_label("segmentation", overwrite=True)
    label_data = np.zeros((1, 64, 64), dtype=np.uint32)  # zyx
    y0, x0 = 20 + _SHIFT_Y_PX, 20 + _SHIFT_X_PX
    label_data[0, y0 : y0 + 10, x0 : x0 + 10] = 1
    label.set_array(label_data)
    label.consolidate()

    # add masking roi table
    masking_roi_table = label.build_masking_roi_table()
    ome1_pre.add_table("masking_roi_table", masking_roi_table, backend=_TABLE_BACKEND)

    _run_full_pipeline(zarr_url_0, zarr_url_1, overwrite_input=True)

    ome1 = open_ome_zarr_container(zarr_url_1)
    assert "segmentation" in ome1.list_labels(), "Label not copied to registered image"
    registered_label = ome1.get_label("segmentation")
    assert registered_label.get_array().any(), "Registered label is all zeros"

    # Masking ROI table is copied from the acquisition unchanged.
    # apply uses list_tables(filter_types="roi_table") so masking ROI tables
    # are treated as acquisition-specific tables, not reference ROI tables.
    assert "masking_roi_table" in ome1.list_tables()
    masking_table = ome1.get_table("masking_roi_table")
    assert len(masking_table.dataframe) > 0


# ---------------------------------------------------------------------------
# Axes support — calculate_registration_image_based
# ---------------------------------------------------------------------------
#
# ngio handles squeezed/implicit channels transparently (get_channel_idx works
# even without an explicit 'c' axis) and reports is_time_series=False for t=1,
# so the task already supports all axes combinations below without code changes.
# ---------------------------------------------------------------------------

_AXES_PARAMS = [
    ("yx", (64, 64)),
    ("zyx", (4, 64, 64)),
    ("cyx", (1, 64, 64)),
    ("tyx", (1, 64, 64)),  # t=1: must succeed after Phase 2
    ("tzyx", (1, 4, 64, 64)),  # t=1: must succeed after Phase 2
]


@pytest.mark.parametrize("axes,shape", _AXES_PARAMS)
def test_calculate_registration_axes_support(
    tmp_path: Path, axes: str, shape: tuple[int, ...]
):
    """calculate_registration_image_based works for various axes configurations.

    Covers yx, zyx, cyx (no z), and single-timepoint tyx / tzyx.
    ngio handles squeezed channels and t=1 transparently, so no task-code
    changes are needed for any of these axes.
    """
    urls = _build_multiplex_plate(tmp_path, axes, shape)

    calculate_registration_image_based(
        zarr_url=urls["zarr_url_1"],
        init_args=InitArgsRegistration(reference_zarr_url=urls["zarr_url_0"]),
        wavelength_id=_WAVELENGTH,
        roi_table="FOV_ROI_table",
        level=2,
    )

    ome1 = open_ome_zarr_container(urls["zarr_url_1"])
    rois = ome1.get_generic_roi_table("FOV_ROI_table").rois()
    assert len(rois) == 1
    roi = rois[0]
    assert roi.model_extra is not None
    assert roi.model_extra["translation_y"] == pytest.approx(-_SHIFT_Y_UM, abs=0.1)
    assert roi.model_extra["translation_x"] == pytest.approx(-_SHIFT_X_UM, abs=0.1)


# ---------------------------------------------------------------------------
# Full pipeline (calculate → consensus → apply) — all axes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("axes,shape", _AXES_PARAMS)
def test_full_pipeline_axes_support(tmp_path: Path, axes: str, shape: tuple[int, ...]):
    """Full pipeline works end-to-end for every supported axes configuration."""
    urls = _build_multiplex_plate(tmp_path, axes, shape)
    zarr_url_0 = urls["zarr_url_0"]
    zarr_url_1 = urls["zarr_url_1"]

    result = _run_full_pipeline(zarr_url_0, zarr_url_1, overwrite_input=True)

    assert result == {"image_list_updates": [{"zarr_url": zarr_url_1}]}
    assert Path(zarr_url_1).exists()

    ome1 = open_ome_zarr_container(zarr_url_1)
    img1 = ome1.get_image()
    assert img1.get_array().any(), f"Registered image is all zeros for axes={axes}"


# ---------------------------------------------------------------------------
# Time-series guard — tyx with t > 1 must raise
# ---------------------------------------------------------------------------


def test_calculate_registration_tyx_t_gt1_raises(tmp_path: Path):
    """Time-series images with t > 1 are not supported and must raise ValueError."""
    zarr_url = str(tmp_path / "img_tyx_t2.zarr")
    ome = create_empty_ome_zarr(
        zarr_url,
        shape=(2, 64, 64),  # t=2, yx
        xy_pixelsize=_PIXELSIZE,
        axes_names="tyx",
        levels=_LEVELS,
        channel_wavelengths=[_WAVELENGTH],
        overwrite=True,
    )
    fov = ome.build_image_roi_table("image")
    ome.add_table("FOV_ROI_table", fov, backend=_TABLE_BACKEND)

    with pytest.raises(ValueError, match="[Tt]ime"):
        calculate_registration_image_based(
            zarr_url=zarr_url,
            init_args=InitArgsRegistration(reference_zarr_url=zarr_url),
            wavelength_id=_WAVELENGTH,
            roi_table="FOV_ROI_table",
            level=2,
        )

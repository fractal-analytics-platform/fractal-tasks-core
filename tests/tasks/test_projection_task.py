from pathlib import Path

import pytest
from ngio import create_empty_ome_zarr, open_ome_zarr_container
from ngio.tables import RoiTable
from ngio.utils import NgioFileExistsError

from fractal_tasks_core._projection_utils import DaskProjectionMethod
from fractal_tasks_core.projection import projection

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_zarr(tmp_path: Path, shape: tuple, axes: str, name: str = "image") -> Path:
    store = tmp_path / f"{name}.zarr"
    ome = create_empty_ome_zarr(
        store=store,
        shape=shape,
        pixelsize=0.1,
        z_spacing=0.5,
        overwrite=False,
        axes_names=axes,
    )
    table = ome.build_image_roi_table("image")
    ome.add_table("well_ROI_table", table, backend="anndata")
    return store


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_projection_basic(tmp_path: Path) -> None:
    """Basic ZYX projection: output URL, shape, pixel_size.z, ROI table."""
    store = _make_zarr(tmp_path, shape=(16, 32, 32), axes="zyx")

    result = projection(zarr_url=str(store))

    updates = result["image_list_updates"]
    assert len(updates) == 1
    zarr_url = updates[0]["zarr_url"]
    origin_url = updates[0]["origin"]
    assert updates[0]["types"] == {"is_3D": False}
    assert updates[0]["attributes"] == {}

    # Output URL is derived from the input with the method suffix
    assert zarr_url == str(tmp_path / "image_mip.zarr")
    assert origin_url == str(store)
    assert Path(zarr_url).exists()

    out = open_ome_zarr_container(zarr_url)
    img = out.get_image()
    assert img.shape == (1, 32, 32)
    assert img.pixel_size.z == 1.0


@pytest.mark.parametrize(
    "shape, axes, expected_shape",
    [
        ((16, 32, 32), "zyx", (1, 32, 32)),
        ((3, 16, 32, 32), "czyx", (3, 1, 32, 32)),
        ((4, 3, 16, 32, 32), "tczyx", (4, 3, 1, 32, 32)),
    ],
)
def test_projection_all_axes(
    shape: tuple, axes: str, expected_shape: tuple, tmp_path: Path
) -> None:
    """Output z-dimension is collapsed to 1 regardless of axis order."""
    store = _make_zarr(tmp_path, shape=shape, axes=axes)

    result = projection(zarr_url=str(store))

    zarr_url = result["image_list_updates"][0]["zarr_url"]
    img = open_ome_zarr_container(zarr_url).get_image()
    assert img.shape == expected_shape


@pytest.mark.parametrize(
    "method",
    [
        DaskProjectionMethod.MIP,
        DaskProjectionMethod.MINIP,
        DaskProjectionMethod.MEANIP,
        DaskProjectionMethod.SUMIP,
    ],
)
def test_projection_all_methods(method: DaskProjectionMethod, tmp_path: Path) -> None:
    """Each projection method produces a correctly-named output zarr."""
    store = _make_zarr(tmp_path, shape=(16, 32, 32), axes="zyx")

    result = projection(zarr_url=str(store), method=method)

    zarr_url = result["image_list_updates"][0]["zarr_url"]
    assert zarr_url == str(tmp_path / f"image_{method.value}.zarr")
    assert Path(zarr_url).exists()
    img = open_ome_zarr_container(zarr_url).get_image()
    assert img.shape == (1, 32, 32)


def test_projection_overwrite(tmp_path: Path) -> None:
    """overwrite=True allows re-running; overwrite=False raises when output exists."""
    store = _make_zarr(tmp_path, shape=(16, 32, 32), axes="zyx")

    # First run creates the output
    projection(zarr_url=str(store), overwrite=True)
    # Second run with overwrite=True must succeed
    projection(zarr_url=str(store), overwrite=True)

    # Running with overwrite=False when output already exists must raise
    with pytest.raises(NgioFileExistsError):
        projection(zarr_url=str(store), overwrite=False)


def test_projection_non_zarr_url_fails(tmp_path: Path) -> None:
    """Input URL that does not end with .zarr raises ValueError."""
    store = _make_zarr(tmp_path, shape=(16, 32, 32), axes="zyx")
    bad_url = str(store).removesuffix(".zarr")

    with pytest.raises(ValueError, match="must end with .zarr"):
        projection(zarr_url=bad_url)


def test_projection_2d_fails(tmp_path: Path) -> None:
    """Passing a 2D image (no z-axis) raises ValueError."""
    store = tmp_path / "image_2d.zarr"
    create_empty_ome_zarr(
        store=store,
        shape=(32, 32),
        pixelsize=0.1,
        overwrite=False,
        axes_names="yx",
    )

    with pytest.raises(ValueError):
        projection(zarr_url=str(store))


def test_projection_roi_table_z_update(tmp_path: Path) -> None:
    """After projection the ROI table's z-slice is updated to (0, 1)."""
    store = _make_zarr(tmp_path, shape=(16, 32, 32), axes="zyx")

    result = projection(zarr_url=str(store))

    zarr_url = result["image_list_updates"][0]["zarr_url"]
    out = open_ome_zarr_container(zarr_url)
    assert "well_ROI_table" in out.list_tables()

    table = out.get_table("well_ROI_table")
    assert isinstance(table, RoiTable)
    z_slice = table.get("image").get("z")
    assert z_slice is not None
    assert z_slice.start == 0
    assert z_slice.length == 1

from pathlib import Path

import pytest
from ngio import (
    ImageInWellPath,
    create_empty_ome_zarr,
    create_empty_plate,
    open_ome_zarr_container,
)

from fractal_tasks_core.tasks.import_ome_zarr import (
    import_ome_zarr,
    open_unknown_container,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_plate(
    tmp_path: Path,
    axes: str = "czyx",
    shape: tuple[int, ...] = (2, 4, 8, 8),
) -> Path:
    """Create a plate with two images (A/01/0, B/02/0) and no ROI tables."""
    images = [
        ImageInWellPath(row="A", column="01", path="0"),
        ImageInWellPath(row="B", column="02", path="0"),
    ]
    plate_path = tmp_path / "plate.zarr"
    plate = create_empty_plate(
        store=plate_path,
        name="plate.zarr",
        images=images,
        overwrite=True,
    )
    for image_path in plate.images_paths():
        row, col, path = image_path.split("/")
        create_empty_ome_zarr(
            store=plate.get_image_store(row, col, path),
            shape=shape,
            xy_pixelsize=0.5,
            axes_names=axes,
            overwrite=True,
            levels=2,
        )
    return plate_path


def _make_image(
    tmp_path: Path,
    axes: str = "czyx",
    shape: tuple[int, ...] = (2, 4, 8, 8),
) -> Path:
    """Create a standalone image zarr with no ROI tables."""
    image_path = tmp_path / "image.zarr"
    create_empty_ome_zarr(
        store=image_path,
        shape=shape,
        xy_pixelsize=0.5,
        axes_names=axes,
        overwrite=True,
        levels=2,
    )
    return image_path


def _check_roi_tables(image_url: str) -> None:
    """Assert both image_ROI_table and grid_ROI_table are present."""
    tables = open_ome_zarr_container(image_url).list_tables()
    assert "image_ROI_table" in tables
    assert "grid_ROI_table" in tables


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "axes, shape, expected_is_3D",
    [
        ("czyx", (2, 4, 8, 8), True),
        ("yx", (8, 8), False),
    ],
)
def test_import_plate(
    tmp_path: Path,
    axes: str,
    shape: tuple[int, ...],
    expected_is_3D: bool,
) -> None:
    _make_plate(tmp_path, axes=axes, shape=shape)

    result = import_ome_zarr(
        zarr_dir=str(tmp_path),
        zarr_name="plate.zarr",
        grid_y_shape=2,
        grid_x_shape=2,
        update_omero_metadata=False,
    )

    updates = result["image_list_updates"]
    assert len(updates) == 2

    well_names = {u["attributes"]["well"] for u in updates}
    assert well_names == {"A01", "B02"}
    for update in updates:
        assert update["attributes"]["plate"] == "plate.zarr"
        assert update["types"]["is_3D"] == expected_is_3D
        _check_roi_tables(update["zarr_url"])


def test_import_well(tmp_path: Path) -> None:
    _make_plate(tmp_path)

    result = import_ome_zarr(
        zarr_dir=str(tmp_path),
        zarr_name="plate.zarr/A/01",
        update_omero_metadata=False,
    )

    updates = result["image_list_updates"]
    assert len(updates) == 1
    assert updates[0]["attributes"] == {"well": "A01"}
    assert "plate" not in updates[0]["attributes"]
    _check_roi_tables(updates[0]["zarr_url"])


def test_import_image(tmp_path: Path) -> None:
    _make_plate(tmp_path)

    result = import_ome_zarr(
        zarr_dir=str(tmp_path),
        zarr_name="plate.zarr/A/01/0",
        update_omero_metadata=False,
    )

    updates = result["image_list_updates"]
    assert len(updates) == 1
    assert "attributes" not in updates[0]
    _check_roi_tables(updates[0]["zarr_url"])


def test_import_roi_tables_disabled(tmp_path: Path) -> None:
    _make_plate(tmp_path)

    result = import_ome_zarr(
        zarr_dir=str(tmp_path),
        zarr_name="plate.zarr",
        add_image_ROI_table=False,
        add_grid_ROI_table=False,
        update_omero_metadata=False,
    )

    for update in result["image_list_updates"]:
        tables = open_ome_zarr_container(update["zarr_url"]).list_tables()
        assert "image_ROI_table" not in tables
        assert "grid_ROI_table" not in tables


@pytest.mark.parametrize(
    "shape, axes, grid_y, grid_x, expected_n_rois",
    [
        ((8, 8), "yx", 2, 2, 16),  # 4 steps in y × 4 in x
        ((8, 8), "yx", 4, 4, 4),  # 2 steps in y × 2 in x
        ((4, 8, 16), "zyx", 4, 4, 8),  # 2 steps in y × 4 in x
    ],
)
def test_grid_roi_table_roi_count(
    tmp_path: Path,
    shape: tuple[int, ...],
    axes: str,
    grid_y: int,
    grid_x: int,
    expected_n_rois: int,
) -> None:
    image_path = _make_image(tmp_path, axes=axes, shape=shape)

    import_ome_zarr(
        zarr_dir=str(tmp_path),
        zarr_name="image.zarr",
        grid_y_shape=grid_y,
        grid_x_shape=grid_x,
        add_image_ROI_table=False,
        update_omero_metadata=False,
    )

    ome_zarr = open_ome_zarr_container(str(image_path))
    grid_table = ome_zarr.get_table("grid_ROI_table")
    assert len(grid_table.rois()) == expected_n_rois


def test_import_omero_metadata_update(tmp_path: Path) -> None:
    n_channels = 3
    _make_image(tmp_path, axes="czyx", shape=(n_channels, 4, 8, 8))

    import_ome_zarr(
        zarr_dir=str(tmp_path),
        zarr_name="image.zarr",
        update_omero_metadata=True,
        add_image_ROI_table=False,
        add_grid_ROI_table=False,
    )

    image = open_ome_zarr_container(str(tmp_path / "image.zarr")).get_image()
    assert len(image.channel_labels) == n_channels


def test_open_unknown_container_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Could not detect OME-NGFF type"):
        open_unknown_container(str(tmp_path / "nonexistent.zarr"))

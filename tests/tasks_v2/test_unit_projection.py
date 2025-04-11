from pathlib import Path

import pytest
from ngio import create_empty_ome_zarr
from ngio import open_ome_zarr_container
from ngio.utils import NgioFileExistsError

from fractal_tasks_core.tasks.projection import InitArgsMIP
from fractal_tasks_core.tasks.projection import projection


@pytest.mark.parametrize(
    "shape, axes, expected_shape",
    [
        ((16, 32, 32), "zyx", (1, 32, 32)),
        ((4, 3, 16, 32, 32), "tczyx", (4, 3, 1, 32, 32)),
        ((1, 1, 16, 32, 32), "tczyx", (1, 1, 1, 32, 32)),
        ((3, 16, 32, 32), "czyx", (3, 1, 32, 32)),
    ],
)
def test_projection(
    shape, axes: str, expected_shape: tuple[int, ...], tmp_path: Path
) -> None:
    """
    Test the projection task.
    """
    # Create a plate with 2 wells and 1 acquisition
    store = tmp_path / "sample_ome_zarr.zarr"
    origin_ome_zarr = create_empty_ome_zarr(
        store=store,
        shape=shape,
        xy_pixelsize=0.1,
        z_spacing=0.5,
        overwrite=False,
        axes_names=axes,
    )
    table = origin_ome_zarr.build_image_roi_table("image")
    origin_ome_zarr.add_table(
        "well_ROI_table", table, backend="experimental_json_v1"
    )

    init_mip = InitArgsMIP(
        origin_url=str(store),
        method="mip",
        overwrite=False,
        new_plate_name="new_plate.zarr",
    )

    mip_store = tmp_path / "sample_ome_zarr_mip.zarr"
    update_list = projection(zarr_url=str(mip_store), init_args=init_mip)

    zarr_url = update_list["image_list_updates"][0]["zarr_url"]
    origin_url = update_list["image_list_updates"][0]["origin"]
    attributes = update_list["image_list_updates"][0]["attributes"]
    types = update_list["image_list_updates"][0]["types"]

    assert Path(zarr_url).exists()
    assert Path(origin_url).exists()
    assert attributes == {"plate": "new_plate.zarr"}
    assert types == {"is_3D": False}

    ome_zarr = open_ome_zarr_container(zarr_url)

    image = ome_zarr.get_image()
    assert image.shape == expected_shape
    assert image.pixel_size.z == 1.0

    assert ome_zarr.list_tables() == ["well_ROI_table"]

    mip_table = ome_zarr.get_table("well_ROI_table", check_type="roi_table")
    assert mip_table.get("image").z_length == 1
    assert mip_table.get("image").z == 0


@pytest.mark.parametrize(
    "shape, axes",
    [
        ((1, 32, 32), "zyx"),
        ((32, 32), "yx"),
        ((4, 3, 1, 32, 32), "tczyx"),
        ((4, 32, 32), "tyx"),
    ],
)
def test_fail_non_3d_projection(shape, axes: str, tmp_path: Path) -> None:
    """
    Test the projection task.
    """
    # Create a plate with 2 wells and 1 acquisition
    store = tmp_path / "sample_ome_zarr.zarr"
    create_empty_ome_zarr(
        store=store,
        shape=shape,
        xy_pixelsize=0.1,
        z_spacing=0.5,
        overwrite=False,
        axes_names=axes,
    )

    init_mip = InitArgsMIP(
        origin_url=str(store),
        method="mip",
        overwrite=False,
        new_plate_name="new_plate.zarr",
    )

    mip_store = tmp_path / "sample_ome_zarr_mip.zarr"
    with pytest.raises(ValueError):
        projection(zarr_url=str(mip_store), init_args=init_mip)


@pytest.mark.parametrize("method", ["mip", "minip", "meanip", "sumip"])
def test_projections_methods(
    sample_ome_zarr_zyx_url: Path, tmp_path: Path, method: str
) -> None:
    """
    Test the projection task.
    """
    init_mip = InitArgsMIP(
        origin_url=str(sample_ome_zarr_zyx_url),
        method=method,
        overwrite=False,
        new_plate_name="new_plate.zarr",
    )

    mip_store = tmp_path / "sample_ome_zarr_mip.zarr"
    update_list = projection(zarr_url=str(mip_store), init_args=init_mip)

    zarr_url = update_list["image_list_updates"][0]["zarr_url"]
    origin_url = update_list["image_list_updates"][0]["origin"]
    attributes = update_list["image_list_updates"][0]["attributes"]
    types = update_list["image_list_updates"][0]["types"]

    assert Path(zarr_url).exists()
    assert Path(origin_url).exists()
    assert attributes == {"plate": "new_plate.zarr"}
    assert types == {"is_3D": False}

    ome_zarr = open_ome_zarr_container(zarr_url)

    image = ome_zarr.get_image()
    assert image.shape == (1, 32, 32)
    assert image.pixel_size.z == 1.0


def test_projection_overwrite(
    sample_ome_zarr_zyx_url: Path, tmp_path: Path
) -> None:
    """
    Test the projection task.
    """
    # Create a plate with 2 wells and 1 acquisition
    init_mip = InitArgsMIP(
        origin_url=str(sample_ome_zarr_zyx_url),
        method="mip",
        overwrite=True,
        new_plate_name="new_plate.zarr",
    )

    mip_store = tmp_path / "sample_ome_zarr_mip.zarr"
    projection(zarr_url=str(mip_store), init_args=init_mip)

    projection(zarr_url=str(mip_store), init_args=init_mip)

    # Check if the overwrite behavior is correct
    init_mip = InitArgsMIP(
        origin_url=str(sample_ome_zarr_zyx_url),
        method="mip",
        overwrite=False,
        new_plate_name="new_plate.zarr",
    )

    with pytest.raises(NgioFileExistsError):
        projection(zarr_url=str(mip_store), init_args=init_mip)

from pathlib import Path

import numpy as np
import pytest
from ngio import create_empty_ome_zarr
from ngio import open_ome_zarr_container
from ngio import PixelSize
from ngio.utils import NgioFileExistsError

from fractal_tasks_core.tasks.io_models import AdvancedArgsMIP
from fractal_tasks_core.tasks.projection import InitArgsMIP
from fractal_tasks_core.tasks.projection import projection


@pytest.mark.parametrize(
    "projection_axis, axes, shape, expected_shape",
    [
        ("z", "zyx", (16, 32, 32), (1, 32, 32)),
        ("z", "tczyx", (4, 3, 16, 32, 32), (4, 3, 1, 32, 32)),
        ("z", "tczyx", (1, 1, 16, 32, 32), (1, 1, 1, 32, 32)),
        ("z", "czyx", (3, 16, 32, 32), (3, 1, 32, 32)),
        ("y", "zyx", (16, 32, 32), (1, 32, 16)),
        ("y", "tczyx", (4, 3, 16, 32, 32), (4, 3, 1, 32, 16)),
        ("y", "tczyx", (1, 1, 16, 32, 32), (1, 1, 1, 32, 16)),
        ("y", "czyx", (3, 16, 32, 32), (3, 1, 32, 16)),
        ("x", "zyx", (16, 32, 32), (1, 16, 32)),
        ("x", "tczyx", (4, 3, 16, 32, 32), (4, 3, 1, 16, 32)),
        ("x", "tczyx", (1, 1, 16, 32, 32), (1, 1, 1, 16, 32)),
        ("x", "czyx", (3, 16, 32, 32), (3, 1, 16, 32)),
        # also for y!=x shapes
        ("z", "zyx", (16, 30, 42), (1, 30, 42)),
        ("z", "tczyx", (4, 3, 16, 30, 42), (4, 3, 1, 30, 42)),
        ("y", "zyx", (16, 30, 42), (1, 42, 16)),
        ("y", "tczyx", (4, 3, 16, 30, 42), (4, 3, 1, 42, 16)),
        ("x", "zyx", (16, 30, 42), (1, 16, 30)),
        ("x", "tczyx", (4, 3, 16, 30, 42), (4, 3, 1, 16, 30)),
    ],
)
def test_3d_projection(
    projection_axis: str,
    axes: str,
    shape,
    expected_shape: tuple[int, ...],
    tmp_path: Path,
) -> None:
    """
    Test the projection task.
    """
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
        advanced_parameters=AdvancedArgsMIP(projection_axis=projection_axis),
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
    if projection_axis == "z":
        assert image.pixel_size.z == 1.0

    # no additional tables should be created
    assert ome_zarr.list_tables() == ["well_ROI_table"]

    # test that the ROIs have been projected correctly
    mip_table = ome_zarr.get_generic_roi_table("well_ROI_table")
    assert mip_table.get("image").z_length == 1
    assert mip_table.get("image").z == 0

    # assert False


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
        advanced_parameters=AdvancedArgsMIP(),
        overwrite=False,
        new_plate_name="new_plate.zarr",
    )

    mip_store = tmp_path / "sample_ome_zarr_mip.zarr"
    with pytest.raises(ValueError, match="The input image is 2D"):
        projection(zarr_url=str(mip_store), init_args=init_mip)


@pytest.mark.parametrize(
    "method, expected_value",
    [
        ("mip", 15),
        ("minip", 0),
        ("meanip", 7),
        ("sumip", 120),
    ],
)
def test_projections_methods(
    tmp_path: Path, method: str, expected_value: int
) -> None:
    """
    Test the projection task with respect to different methods.
    """
    store = tmp_path / "sample_ome_zarr.zarr"
    origin_ome_zarr = create_empty_ome_zarr(
        store=store,
        shape=(16, 32, 32),
        xy_pixelsize=0.1,
        z_spacing=0.5,
        overwrite=False,
        axes_names="zyx",
    )
    table = origin_ome_zarr.build_image_roi_table("image")
    origin_ome_zarr.add_table(
        "well_ROI_table", table, backend="experimental_json_v1"
    )
    origin_image = origin_ome_zarr.get_image()

    # create artificial data with linear increase along z
    input_array = np.zeros(origin_image.shape, dtype=origin_image.dtype)
    for z in range(16):
        input_array[z, :, :] = z
    origin_image.set_array(input_array)

    init_mip = InitArgsMIP(
        origin_url=str(store),
        method=method,
        advanced_parameters=AdvancedArgsMIP(),
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

    image_array = image.get_as_numpy()
    np.testing.assert_array_equal(
        image_array, np.ones((1, 32, 32)) * expected_value
    )


def test_projection_overwrite(
    sample_ome_zarr_zyx_url: Path, tmp_path: Path
) -> None:
    """
    Test the projection task overwrite behavior.
    """
    # Create a plate with 2 wells and 1 acquisition
    init_mip = InitArgsMIP(
        origin_url=str(sample_ome_zarr_zyx_url),
        method="mip",
        advanced_parameters=AdvancedArgsMIP(),
        overwrite=True,
        new_plate_name="new_plate.zarr",
    )

    mip_store = tmp_path / "new_plate.zarr"
    projection(zarr_url=str(mip_store), init_args=init_mip)

    projection(zarr_url=str(mip_store), init_args=init_mip)

    # Check if the overwrite behavior is correct
    init_mip = InitArgsMIP(
        origin_url=str(sample_ome_zarr_zyx_url),
        method="mip",
        advanced_parameters=AdvancedArgsMIP(),
        overwrite=False,
        new_plate_name="new_plate_no_overwrite.zarr",
    )

    mip_store = tmp_path / "new_plate_no_overwrite.zarr"
    projection(zarr_url=str(mip_store), init_args=init_mip)

    with pytest.raises(NgioFileExistsError):
        projection(zarr_url=str(mip_store), init_args=init_mip)


def test_projection_output_pyramid_scaling_and_spacing(tmp_path: Path) -> None:
    """
    Test the projection task output pyramid scaling factors.
    """

    sample_ome_zarr_zyx_url = tmp_path / "sample_ome_zarr_zyx.zarr"
    input_ome_zarr = create_empty_ome_zarr(
        store=sample_ome_zarr_zyx_url,
        shape=(3, 50, 300, 400),
        xy_pixelsize=0.2,
        z_spacing=0.5,
        time_spacing=0.7,
        xy_scaling_factor=2.5,
        z_scaling_factor=2,  # test also this scaling
        overwrite=False,
        space_unit="foot",
        time_unit="day",
        axes_names="tzyx",
    )
    input_meta = input_ome_zarr.image_meta

    expected_pixel_size = {
        "z": PixelSize(
            x=0.2, y=0.2, z=1.0, t=0.7, space_unit="foot", time_unit="day"
        ),
        "y": PixelSize(
            x=0.5, y=0.2, z=1.0, t=0.7, space_unit="foot", time_unit="day"
        ),
        "x": PixelSize(
            x=0.2, y=0.5, z=1.0, t=0.7, space_unit="foot", time_unit="day"
        ),
    }
    expected_yx_scaling = {
        "z": (2.5, 2.5),
        "y": (2.5, 2.0),
        "x": (2.0, 2.5),
    }
    output_stores = {
        "z": tmp_path / "output_mip.zarr",
        "y": tmp_path / "output_mip_y.zarr",
        "x": tmp_path / "output_mip_x.zarr",
    }

    for projection_axis in ["z", "y", "x"]:
        init_mip = InitArgsMIP(
            origin_url=str(sample_ome_zarr_zyx_url),
            method="mip",
            overwrite=False,
            advanced_parameters=AdvancedArgsMIP(
                projection_axis=projection_axis
            ),
            new_plate_name="new_plate.zarr",
        )

        projection(
            zarr_url=str(output_stores[projection_axis]), init_args=init_mip
        )

        ome_zarr_mip = open_ome_zarr_container(output_stores[projection_axis])
        meta_mip = ome_zarr_mip.image_meta
        image_mip = ome_zarr_mip.get_image()

        assert input_meta.levels == meta_mip.levels
        assert meta_mip.z_scaling() == 1
        assert meta_mip.yx_scaling() == expected_yx_scaling[projection_axis]
        assert image_mip.pixel_size == expected_pixel_size[projection_axis], (
            f"Failed for {projection_axis=}"
            f"\nexpected {expected_pixel_size[projection_axis]}"
            f"\ngot      {image_mip.pixel_size}"
        )


@pytest.mark.parametrize("projection_axis", ["z", "y", "x"])
@pytest.mark.parametrize(
    "autofocus_radius, focused_plane, expected_sum",
    [
        (5, 7, 11),  # normal case
        (0, 4, 1),  # zero radius - only take the sharpest plane
        (1, 4, 3),  # smallest non-zero radius
        (4, 1, 6),  # focus near the edge
        (4, 15, 5),  # focus near the other edge
    ],
)
def test_projection_autofocus(
    tmp_path: Path,
    projection_axis: str,
    autofocus_radius: int,
    focused_plane: int,
    expected_sum: int,
) -> None:
    """
    Test the projection task with autofocus. We test with sumip method,
    as it enables to verify how many planes contributed in the projection.
    """
    store = tmp_path / "sample_ome_zarr.zarr"
    origin_ome_zarr = create_empty_ome_zarr(
        store=store,
        shape=(2, 16, 16, 16),
        xy_pixelsize=0.1,
        z_spacing=0.5,
        overwrite=False,
        chunks=(1, 4, 4, 4),  # small chunks to test slicing in autofocus
        axes_names="czyx",
    )
    table = origin_ome_zarr.build_image_roi_table("image")
    origin_ome_zarr.add_table(
        "well_ROI_table", table, backend="experimental_json_v1"
    )
    origin_image = origin_ome_zarr.get_image()

    # create artificial data with a sharpest plane at index 8
    # regardless of channel and projection axis
    input_array = np.ones(origin_image.shape, dtype=origin_image.dtype)
    input_array[:, focused_plane, focused_plane, focused_plane] = 10.0
    origin_image.set_array(input_array)

    init_mip_af = InitArgsMIP(
        origin_url=str(store),
        method="sumip",
        overwrite=False,
        advanced_parameters=AdvancedArgsMIP(
            projection_axis=projection_axis,
            autofocus_radius=autofocus_radius,
        ),
        new_plate_name="new_plate_autofocus.zarr",
    )

    # get projection with autofocus
    mip_store = tmp_path / "new_plate_autofocus.zarr"
    update_list = projection(zarr_url=str(mip_store), init_args=init_mip_af)
    zarr_url = update_list["image_list_updates"][0]["zarr_url"]
    ome_zarr = open_ome_zarr_container(zarr_url)
    image_autofocus = ome_zarr.get_image()
    array_autofocus = image_autofocus.get_as_numpy()

    expected_output = np.ones_like(array_autofocus) * expected_sum
    # the sharpest plane contributes with value 10
    expected_output[:, :, focused_plane, focused_plane] += 9
    np.testing.assert_array_equal(array_autofocus, expected_output)


@pytest.mark.parametrize(
    "projection_axis, upscale_factor, z_interpolation_order, expected_shape",
    [
        ("y", 4.0, 1, (1, 32, 64)),
        ("x", 4.0, 2, (1, 64, 32)),
        # non-integer factor
        ("y", 1.5, 3, (1, 32, 24)),
        ("x", 1.5, 4, (1, 24, 32)),
    ],
)
def test_projection_z_upscale_factor(
    sample_ome_zarr_zyx_url: Path,
    tmp_path: Path,
    projection_axis: str,
    upscale_factor: float,
    z_interpolation_order: int,
    expected_shape: tuple[int, int, int],
) -> None:
    """
    Test the projection task with z upscaling after projection.
    """

    init_mip = InitArgsMIP(
        origin_url=str(sample_ome_zarr_zyx_url),
        method="mip",
        overwrite=True,
        advanced_parameters=AdvancedArgsMIP(
            projection_axis=projection_axis,
            z_upscale_factor=upscale_factor,
            z_upscale_interpolation_order=z_interpolation_order,
        ),
        new_plate_name="new_plate_z_upscale.zarr",
    )

    mip_store = tmp_path / "new_plate_z_upscale.zarr"
    update_list = projection(zarr_url=str(mip_store), init_args=init_mip)
    zarr_url = update_list["image_list_updates"][0]["zarr_url"]
    ome_zarr = open_ome_zarr_container(zarr_url)
    image = ome_zarr.get_image()

    assert image.shape == expected_shape


def test_projection_z_upscale_factor_invalid(
    sample_ome_zarr_zyx_url: Path,
    tmp_path: Path,
) -> None:
    """
    Test the projection task with invalid z upscaling.
    """

    # projection with respect to z axis cannot have z upscaling
    with pytest.raises(ValueError, match="z_upscale_factor can only be used"):
        init_mip = InitArgsMIP(
            origin_url=str(sample_ome_zarr_zyx_url),
            method="mip",
            overwrite=True,
            advanced_parameters=AdvancedArgsMIP(
                projection_axis="z",
                z_upscale_factor=4.0,
                z_upscale_interpolation_order=3,
            ),
            new_plate_name="new_plate_z_upscale.zarr",
        )

        mip_store = tmp_path / "new_plate_z_upscale.zarr"
        projection(zarr_url=str(mip_store), init_args=init_mip)

    # projection with negative z upscaling factor
    with pytest.raises(ValueError, match="z_upscale_factor must be"):
        init_mip = InitArgsMIP(
            origin_url=str(sample_ome_zarr_zyx_url),
            method="mip",
            overwrite=True,
            advanced_parameters=AdvancedArgsMIP(
                projection_axis="y",
                z_upscale_factor=-2.0,
                z_upscale_interpolation_order=3,
            ),
            new_plate_name="new_plate_z_upscale.zarr",
        )

        mip_store = tmp_path / "new_plate_z_upscale.zarr"
        projection(zarr_url=str(mip_store), init_args=init_mip)

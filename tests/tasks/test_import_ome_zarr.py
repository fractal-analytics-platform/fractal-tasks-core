import zarr
from devtools import debug

from .._zenodo_ome_zarrs import prepare_3D_zarr
from fractal_tasks_core.tasks.copy_ome_zarr import copy_ome_zarr
from fractal_tasks_core.tasks.import_ome_zarr import import_ome_zarr
from fractal_tasks_core.tasks.maximum_intensity_projection import (
    maximum_intensity_projection,
)  # noqa


def _check_ROI_tables(_image_path):
    g = zarr.open_group(f"{_image_path}/tables", mode="r")
    debug(g.attrs.asdict())
    assert g.attrs["tables"] == ["image_ROI_table", "grid_ROI_table"]
    zarr.open_group(f"{_image_path}/tables/image_ROI_table", mode="r")
    zarr.open_group(f"{_image_path}/tables/grid_ROI_table", mode="r")


def test_import_ome_zarr_plate(tmp_path, zenodo_zarr, zenodo_zarr_metadata):

    # Prepare an on-disk OME-Zarr at the plate level
    root_path = tmp_path
    prepare_3D_zarr(
        root_path, zenodo_zarr, zenodo_zarr_metadata, remove_tables=True
    )
    zarr_name = "plate.zarr"

    # Run import_ome_zarr
    metadiff = import_ome_zarr(
        input_paths=[str(root_path)],
        zarr_name=zarr_name,
        output_path="null",
        metadata={},
        grid_ROI_shape=(3, 3),
    )
    metadata = metadiff.copy()

    # Check metadata
    EXPECTED_METADATA = dict(
        plate=["plate.zarr"],
        well=["plate.zarr/B/03"],
        image=["plate.zarr/B/03/0"],
    )
    assert metadata == EXPECTED_METADATA

    # Check that table were created
    _check_ROI_tables(f"{root_path}/{zarr_name}/B/03/0")

    # Run copy_ome_zarr and maximum_intensity_projection
    metadata_update = copy_ome_zarr(
        input_paths=[str(root_path)],
        output_path=str(root_path),
        metadata=metadata,
        project_to_2D=True,
        suffix="mip",
        ROI_table_names=("image_ROI_table", "grid_ROI_table"),
    )
    metadata.update(metadata_update)
    debug(metadata)
    for component in metadata["image"]:
        maximum_intensity_projection(
            input_paths=[str(tmp_path)],
            output_path=str(tmp_path),
            metadata=metadata,
            component=component,
        )


def test_import_ome_zarr_well(tmp_path, zenodo_zarr, zenodo_zarr_metadata):

    # Prepare an on-disk OME-Zarr at the plate level
    root_path = tmp_path
    prepare_3D_zarr(
        root_path, zenodo_zarr, zenodo_zarr_metadata, remove_tables=True
    )
    zarr_name = "plate.zarr/B/03"

    # Run import_ome_zarr
    metadiff = import_ome_zarr(
        input_paths=[str(root_path)],
        zarr_name=zarr_name,
        output_path="null",
        metadata={},
        grid_ROI_shape=(3, 3),
    )
    metadata = metadiff.copy()

    # Check metadata
    EXPECTED_METADATA = dict(
        well=["plate.zarr/B/03"],
        image=["plate.zarr/B/03/0"],
    )
    assert metadata == EXPECTED_METADATA

    # Check that table were created
    _check_ROI_tables(f"{root_path}/{zarr_name}/0")


def test_import_ome_zarr_image(tmp_path, zenodo_zarr, zenodo_zarr_metadata):

    # Prepare an on-disk OME-Zarr at the plate level
    root_path = tmp_path
    prepare_3D_zarr(
        root_path, zenodo_zarr, zenodo_zarr_metadata, remove_tables=True
    )
    zarr_name = "plate.zarr/B/03/0"

    # Run import_ome_zarr
    metadiff = import_ome_zarr(
        input_paths=[str(root_path)],
        zarr_name=zarr_name,
        output_path="null",
        metadata={},
        grid_ROI_shape=(3, 3),
    )
    metadata = metadiff.copy()

    # Check metadata
    EXPECTED_METADATA = dict(
        image=["plate.zarr/B/03/0"],
    )
    assert metadata == EXPECTED_METADATA

    # Check that table were created
    _check_ROI_tables(f"{root_path}/{zarr_name}")

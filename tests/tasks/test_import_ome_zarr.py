from devtools import debug

from .._zenodo_ome_zarrs import prepare_3D_zarr
from fractal_tasks_core.tasks.copy_ome_zarr import copy_ome_zarr
from fractal_tasks_core.tasks.import_ome_zarr import import_ome_zarr
from fractal_tasks_core.tasks.maximum_intensity_projection import (
    maximum_intensity_projection,
)  # noqa


def test_import_ome_zarr(tmp_path, zenodo_zarr, zenodo_zarr_metadata):

    # Create a zarr
    prepare_3D_zarr(
        tmp_path, zenodo_zarr, zenodo_zarr_metadata, remove_tables=True
    )

    zarr_path = str(tmp_path / "plate.zarr")
    metadiff = import_ome_zarr(
        input_paths=[zarr_path],
        output_path="null",
        metadata={},
        grid_ROI_shape=(1, 2),
    )
    metadata = metadiff.copy()
    print(zarr_path)

    # ls /tmp/pytest-of-tommaso/pytest-10/test_import_ome_zarr0/plate.zarr/B/03/0/tables/ FOV_ROI_table  grid_ROI_table  image_ROI_table  well_ROI_table  # noqa

    # Replicate
    metadata_update = copy_ome_zarr(
        input_paths=[str(tmp_path)],
        output_path=str(tmp_path),
        metadata=metadata,
        project_to_2D=True,
        suffix="mip",
        ROI_table_names=("image_ROI_table", "grid_ROI_table"),
    )
    metadata.update(metadata_update)
    debug(metadata)

    # MIP
    for component in metadata["image"]:
        maximum_intensity_projection(
            input_paths=[str(tmp_path)],
            output_path=str(tmp_path),
            metadata=metadata,
            component=component,
            ROI_table_name="grid_ROI_table",
        )

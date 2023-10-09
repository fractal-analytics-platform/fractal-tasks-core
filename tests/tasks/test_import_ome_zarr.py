from devtools import debug

from .._zenodo_ome_zarrs import prepare_3D_zarr
from fractal_tasks_core.tasks.import_ome_zarr import import_ome_zarr


def test_import_ome_zarr(tmp_path, zenodo_zarr, zenodo_zarr_metadata):
    prepare_3D_zarr(tmp_path, zenodo_zarr, zenodo_zarr_metadata)
    print(tmp_path)
    plate_path = str(tmp_path / "plate.zarr")
    input_paths = [plate_path]
    output_path = str(tmp_path / "output")
    metadiff = import_ome_zarr(
        input_paths=input_paths,
        output_path=output_path,
        metadata={},
    )
    debug(metadiff)

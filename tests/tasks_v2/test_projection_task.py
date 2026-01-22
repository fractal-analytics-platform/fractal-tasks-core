from pathlib import Path

from ngio import open_ome_zarr_container

from fractal_tasks_core.tasks.copy_ome_zarr_hcs_plate import (
    copy_ome_zarr_hcs_plate,
)
from fractal_tasks_core.tasks.projection import projection


def test_mip_task(cardiomyocyte_tiny_path: Path, tmp_path: Path) -> None:
    image_url = str(cardiomyocyte_tiny_path / "B" / "03" / "0")
    parallel_list = copy_ome_zarr_hcs_plate(
        zarr_urls=[image_url],
        zarr_dir=str(tmp_path / "tmp_out"),
        overwrite=True,
        re_initialize_plate=True,
    )

    assert len(parallel_list["parallelization_list"]) == 1

    image = parallel_list["parallelization_list"][0]
    update_list = projection(**image)

    zarr_url = update_list["image_list_updates"][0]["zarr_url"]
    origin_url = update_list["image_list_updates"][0]["origin"]
    attributes = update_list["image_list_updates"][0]["attributes"]
    types = update_list["image_list_updates"][0]["types"]

    assert Path(zarr_url).exists()
    assert Path(origin_url).exists()
    assert attributes == {
        "plate": "20200812-CardiomyocyteDifferentiation14-Cycle1-tiny_mip.zarr"
    }
    assert types == {"is_3D": False}

    ome_zarr = open_ome_zarr_container(zarr_url)

    image = ome_zarr.get_image()
    assert image.dimensions.get("z", default=None) == 1
    assert image.pixel_size.z == 1.0

    assert ome_zarr.list_tables() == ["FOV_ROI_table", "well_ROI_table"]

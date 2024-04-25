import pytest
import zarr
from devtools import debug

from .._zenodo_ome_zarrs import prepare_3D_zarr
from fractal_tasks_core.tables.v1 import get_tables_list_v1
from fractal_tasks_core.tasks.copy_ome_zarr_hcs_plate import (
    copy_ome_zarr_hcs_plate,
)
from fractal_tasks_core.tasks.import_ome_zarr import import_ome_zarr
from fractal_tasks_core.tasks.maximum_intensity_projection import (
    maximum_intensity_projection,
)

# from fractal_tasks_core.channels import ChannelInputModel


def _check_ROI_tables(_image_path):
    g = zarr.open_group(f"{_image_path}/tables", mode="r")
    debug(g.attrs.asdict())
    assert g.attrs["tables"] == ["image_ROI_table", "grid_ROI_table"]
    zarr.open_group(f"{_image_path}/tables/image_ROI_table", mode="r")
    zarr.open_group(f"{_image_path}/tables/grid_ROI_table", mode="r")


def test_import_ome_zarr_plate(tmp_path, zenodo_zarr):

    # Prepare an on-disk OME-Zarr at the plate level
    zarr_dir = str(tmp_path)
    prepare_3D_zarr(zarr_dir, zenodo_zarr, remove_tables=True)
    zarr_name = "plate.zarr"

    # Run import_ome_zarr
    image_list_changes = import_ome_zarr(
        zarr_urls=[],
        zarr_dir=zarr_dir,
        zarr_name=zarr_name,
        grid_y_shape=3,
        grid_x_shape=3,
    )
    debug(image_list_changes)
    zarr_urls = [
        x["zarr_url"] for x in image_list_changes["image_list_updates"]
    ]

    expected_image_list_changes = {
        "image_list_updates": [
            {
                "zarr_url": zarr_urls[0],
                "attributes": {
                    "plate": "plate.zarr",
                    "well": "B03",
                },
                "types": {
                    "is_3D": True,
                },
            },
        ],
    }
    assert expected_image_list_changes == image_list_changes

    # Check that table were created
    _check_ROI_tables(f"{zarr_dir}/{zarr_name}/B/03/0")

    # Run copy_ome_zarr and maximum_intensity_projection
    parallelization_list = copy_ome_zarr_hcs_plate(
        zarr_urls=zarr_urls,
        zarr_dir="tmp_out",
        overwrite=True,
    )["parallelization_list"]
    debug(parallelization_list)

    for image in parallelization_list:
        maximum_intensity_projection(
            zarr_url=image["zarr_url"],
            init_args=image["init_args"],
            overwrite=True,
        )


def test_import_ome_zarr_well(tmp_path, zenodo_zarr):

    # Prepare an on-disk OME-Zarr at the plate level
    zarr_dir = str(tmp_path)
    prepare_3D_zarr(zarr_dir, zenodo_zarr, remove_tables=True)
    zarr_name = "plate.zarr/B/03"

    # Run import_ome_zarr
    image_list_changes = import_ome_zarr(
        zarr_urls=[],
        zarr_dir=zarr_dir,
        zarr_name=zarr_name,
        grid_y_shape=3,
        grid_x_shape=3,
    )
    debug(image_list_changes)
    zarr_urls = [
        x["zarr_url"] for x in image_list_changes["image_list_updates"]
    ]

    expected_image_list_changes = {
        "image_list_updates": [
            {
                "zarr_url": zarr_urls[0],
                "attributes": {
                    "well": "plate.zarr/B/03",
                },
                "types": {
                    "is_3D": True,
                },
            },
        ],
    }
    assert expected_image_list_changes == image_list_changes

    # Check that table were created
    _check_ROI_tables(f"{zarr_dir}/{zarr_name}/0")


@pytest.mark.parametrize("reset_omero", [True, False])
def test_import_ome_zarr_image(tmp_path, zenodo_zarr, reset_omero):

    # Prepare an on-disk OME-Zarr at the plate level
    zarr_dir = str(tmp_path)
    prepare_3D_zarr(
        zarr_dir,
        zenodo_zarr,
        remove_tables=True,
        remove_omero=reset_omero,
    )
    zarr_name = "plate.zarr/B/03/0"

    # Run import_ome_zarr
    image_list_changes = import_ome_zarr(
        zarr_urls=[],
        zarr_dir=zarr_dir,
        zarr_name=zarr_name,
        grid_y_shape=3,
        grid_x_shape=3,
    )
    debug(image_list_changes)
    zarr_urls = [
        x["zarr_url"] for x in image_list_changes["image_list_updates"]
    ]

    expected_image_list_changes = {
        "image_list_updates": [
            {
                "zarr_url": zarr_urls[0],
                "types": {
                    "is_3D": True,
                },
            },
        ],
    }
    assert expected_image_list_changes == image_list_changes

    # Check that table were created
    _check_ROI_tables(f"{zarr_dir}/{zarr_name}")

    # Check that omero attributes were filled correctly
    g = zarr.open_group(f"{zarr_dir}/{zarr_name}", mode="r")
    debug(g.attrs["omero"]["channels"])
    if reset_omero:
        EXPECTED_CHANNELS = [
            dict(label="1", wavelength_id="1", color="00FFFF")
        ]
        assert g.attrs["omero"]["channels"] == EXPECTED_CHANNELS
    else:
        EXPECTED_LABEL = "DAPI"
        EXPECTED_WAVELENGTH_ID = "A01_C01"
        assert g.attrs["omero"]["channels"][0]["label"] == EXPECTED_LABEL
        assert (
            g.attrs["omero"]["channels"][0]["wavelength_id"]
            == EXPECTED_WAVELENGTH_ID
        )


def test_import_ome_zarr_image_wrong_channels(tmp_path, zenodo_zarr):
    # Prepare an on-disk OME-Zarr at the plate level
    zarr_dir = str(tmp_path)
    prepare_3D_zarr(
        zarr_dir,
        zenodo_zarr,
        remove_tables=True,
        remove_omero=True,
    )
    zarr_name = "plate.zarr/B/03/0"
    # Modify NGFF omero metadata, adding two channels (even if the Zarr array
    # has only one)
    g = zarr.open_group(f"{zarr_dir}/{zarr_name}", mode="r+")
    new_omero = dict(
        channels=[
            dict(color="asd"),
            dict(color="asd"),
        ]
    )
    g.attrs.update(omero=new_omero)
    # Run import_ome_zarr and catch the error
    with pytest.raises(ValueError) as e:
        _ = import_ome_zarr(
            zarr_urls=[],
            zarr_dir=zarr_dir,
            zarr_name=zarr_name,
            grid_y_shape=3,
            grid_x_shape=3,
        )
    debug(e.value)
    assert "Channels-number mismatch" in str(e.value)


def test_import_ome_zarr_plate_no_ROI_tables(tmp_path, zenodo_zarr):

    # Prepare an on-disk OME-Zarr at the plate level
    zarr_dir = str(tmp_path)
    prepare_3D_zarr(zarr_dir, zenodo_zarr, remove_tables=True)
    zarr_name = "plate.zarr"

    # Run import_ome_zarr
    image_list_changes = import_ome_zarr(
        zarr_urls=[],
        zarr_dir=zarr_dir,
        zarr_name=zarr_name,
        add_image_ROI_table=False,
        add_grid_ROI_table=False,
    )
    debug(image_list_changes)
    zarr_urls = [
        x["zarr_url"] for x in image_list_changes["image_list_updates"]
    ]

    expected_image_list_changes = {
        "image_list_updates": [
            {
                "zarr_url": zarr_urls[0],
                "attributes": {
                    "plate": "plate.zarr",
                    "well": "B03",
                },
                "types": {
                    "is_3D": True,
                },
            },
        ],
    }
    assert expected_image_list_changes == image_list_changes

    # Check that no tables were created
    assert not get_tables_list_v1(zarr_urls[0])

    # Run copy_ome_zarr and maximum_intensity_projection
    # to verify that they run without ROI tables
    parallelization_list = copy_ome_zarr_hcs_plate(
        zarr_urls=zarr_urls,
        zarr_dir="tmp_out",
        overwrite=True,
    )["parallelization_list"]
    debug(parallelization_list)

    for image in parallelization_list:
        maximum_intensity_projection(
            zarr_url=image["zarr_url"],
            init_args=image["init_args"],
            overwrite=True,
        )


# @pytest.mark.skip
def test_import_ome_zarr_image_BIA(tmp_path, monkeypatch):
    """
    This test imports one of the BIA OME-Zarr listed in
    https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD843.

    It is currently marked as "skip", to avoid incurring into download-rate
    limits.

    Also note that any further processing of the imported Zarr this will fail
    because we don't support time data, see fractal-tasks-core issue #169.
    """

    from ftplib import FTP
    import zipfile
    import anndata as ad
    import numpy as np

    # Download an existing OME-Zarr from BIA
    ftp = FTP("ftp.ebi.ac.uk")
    ftp.login()
    ftp.cwd("biostudies/fire/S-BIAD/843/S-BIAD843/Files")
    fname = "WD1_15-02_WT_confocalonly.ome.zarr.zip"
    with (tmp_path / fname).open("wb") as fp:
        ftp.retrbinary(f"RETR {fname}", fp.write)

    with zipfile.ZipFile(tmp_path / fname, "r") as zip_ref:
        zip_ref.extractall(tmp_path)

    zarr_dir = str(tmp_path)
    zarr_name = "WD1_15-02_WT_confocalonly.zarr/0"

    # Run import_ome_zarr
    image_list_changes = import_ome_zarr(
        zarr_urls=[],
        zarr_dir=zarr_dir,
        zarr_name=zarr_name,
        grid_y_shape=3,
        grid_x_shape=3,
    )
    debug(image_list_changes)
    # zarr_urls = [
    #     x["zarr_url"] for x in image_list_changes["image_list_updates"]
    # ]

    # Check that table were created
    _check_ROI_tables(f"{zarr_dir}/{zarr_name}")

    # Check image_ROI_table
    g = zarr.open(f"{zarr_dir}/{zarr_name}", mode="r")
    debug(g.attrs.asdict())
    pixel_size_x = g.attrs["multiscales"][0]["datasets"][0][
        "coordinateTransformations"
    ][0]["scale"][
        -1
    ]  # noqa
    debug(pixel_size_x)
    g = zarr.open(f"{zarr_dir}/{zarr_name}/0", mode="r")
    array_shape_x = g.shape[-1]
    debug(array_shape_x)
    EXPECTED_X_LENGTH = array_shape_x * pixel_size_x
    image_ROI_table = ad.read_zarr(
        f"{zarr_dir}/{zarr_name}/tables/image_ROI_table"
    )
    debug(image_ROI_table.X)
    assert np.allclose(
        image_ROI_table[:, "len_x_micrometer"].X[0, 0],
        EXPECTED_X_LENGTH,
    )

    g = zarr.open(f"{zarr_dir}/{zarr_name}", mode="r")
    omero_channels = g.attrs["omero"]["channels"]
    debug(omero_channels)
    assert len(omero_channels) == 1
    omero_channel = omero_channels[0]
    assert omero_channel["label"] == "Channel 0"
    assert omero_channel["wavelength_id"] == "Channel 0"

    # Part 2: run Cellpose on the imported OME-Zarr.

    # Cellpose task deactivated, as it cannot handle t axis of this dataset yet
    # from fractal_tasks_core.tasks.cellpose_segmentation import (
    #     cellpose_segmentation
    # )
    # from .test_workflows_cellpose_segmentation import (
    #     patched_cellpose_core_use_gpu,
    #     patched_segment_ROI,
    # )

    # monkeypatch.setattr(
    #     "fractal_tasks_core.tasks.cellpose_segmentation.cellpose.core.use_gpu",
    #     patched_cellpose_core_use_gpu,
    # )

    # monkeypatch.setattr(
    #     "fractal_tasks_core.tasks.cellpose_segmentation.segment_ROI",
    #     patched_segment_ROI,
    # )

    # # Per-FOV labeling
    # for zarr_url in zarr_urls:
    #     cellpose_segmentation(
    #         zarr_url=zarr_url,
    #         input_ROI_table="grid_ROI_table",
    #         channel=ChannelInputModel(wavelength_id="Channel 0"),
    #         level=0,
    #         relabeling=True,
    #         diameter_level0=80.0,
    #         augment=True,
    #         net_avg=True,
    #         min_size=30,
    #     )

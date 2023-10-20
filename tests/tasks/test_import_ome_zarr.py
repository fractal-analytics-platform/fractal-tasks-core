import pytest
import zarr
from devtools import debug

import fractal_tasks_core.tasks  # noqa
from .._zenodo_ome_zarrs import prepare_3D_zarr
from fractal_tasks_core.lib_input_models import Channel
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
        grid_y_shape=3,
        grid_x_shape=3,
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
        grid_y_shape=3,
        grid_x_shape=3,
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


@pytest.mark.parametrize("reset_omero", [True, False])
def test_import_ome_zarr_image(
    tmp_path, zenodo_zarr, zenodo_zarr_metadata, reset_omero
):

    # Prepare an on-disk OME-Zarr at the plate level
    root_path = tmp_path
    prepare_3D_zarr(
        root_path,
        zenodo_zarr,
        zenodo_zarr_metadata,
        remove_tables=True,
        remove_omero=reset_omero,
    )
    zarr_name = "plate.zarr/B/03/0"

    # Run import_ome_zarr
    metadiff = import_ome_zarr(
        input_paths=[str(root_path)],
        zarr_name=zarr_name,
        output_path="null",
        metadata={},
        grid_y_shape=3,
        grid_x_shape=3,
    )
    metadata = metadiff.copy()

    # Check metadata
    EXPECTED_METADATA = dict(
        image=["plate.zarr/B/03/0"],
    )
    assert metadata == EXPECTED_METADATA

    # Check that table were created
    _check_ROI_tables(f"{root_path}/{zarr_name}")

    # Check that omero attributes were filled correctly
    g = zarr.open_group(str(root_path / zarr_name), mode="r")
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


def test_import_ome_zarr_image_wrong_channels(
    tmp_path, zenodo_zarr, zenodo_zarr_metadata
):
    # Prepare an on-disk OME-Zarr at the plate level
    root_path = tmp_path
    prepare_3D_zarr(
        root_path,
        zenodo_zarr,
        zenodo_zarr_metadata,
        remove_tables=True,
        remove_omero=True,
    )
    zarr_name = "plate.zarr/B/03/0"
    # Modify NGFF omero metadata, adding two channels (even if the Zarr array
    # has only one)
    g = zarr.open_group(str(root_path / zarr_name), mode="r+")
    new_omero = dict(
        channels=[
            dict(color="asd"),
            dict(color="asd"),
        ]
    )
    g.attrs.update(omero=new_omero)
    # Run import_ome_zarr and catch the error
    with pytest.raises(ValueError) as e:
        import_ome_zarr(
            input_paths=[str(root_path)],
            zarr_name=zarr_name,
            output_path="null",
            metadata={},
        )
    debug(e.value)
    assert "Channels-number mismatch" in str(e.value)


@pytest.mark.skip
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

    root_path = str(tmp_path)
    zarr_name = "WD1_15-02_WT_confocalonly.zarr/0"

    # Run import_ome_zarr
    metadiff = import_ome_zarr(
        input_paths=[str(root_path)],
        zarr_name=zarr_name,
        output_path="null",
        metadata={},
    )
    metadata = metadiff.copy()
    debug(metadata)

    # Check that table were created
    _check_ROI_tables(f"{root_path}/{zarr_name}")

    # Check image_ROI_table
    g = zarr.open(f"{root_path}/{zarr_name}", mode="r")
    debug(g.attrs.asdict())
    pixel_size_x = g.attrs["multiscales"][0]["datasets"][0][
        "coordinateTransformations"
    ][0]["scale"][
        -1
    ]  # noqa
    debug(pixel_size_x)
    g = zarr.open(f"{root_path}/{zarr_name}/0", mode="r")
    array_shape_x = g.shape[-1]
    debug(array_shape_x)
    EXPECTED_X_LENGTH = array_shape_x * pixel_size_x
    image_ROI_table = ad.read_zarr(
        f"{root_path}/{zarr_name}/tables/image_ROI_table"
    )
    debug(image_ROI_table.X)
    assert np.allclose(
        image_ROI_table[:, "len_x_micrometer"].X[0, 0],
        EXPECTED_X_LENGTH,
    )

    g = zarr.open(f"{root_path}/{zarr_name}", mode="r")
    omero_channels = g.attrs["omero"]["channels"]
    debug(omero_channels)
    assert len(omero_channels) == 1
    omero_channel = omero_channels[0]
    assert omero_channel["label"] == "Channel 0"
    assert omero_channel["wavelength_id"] == "Channel 0"

    # Part 2: run Cellpose on the imported OME-Zarr.

    from fractal_tasks_core.cellpose_segmentation import cellpose_segmentation
    from .test_workflows_cellpose_segmentation import (
        patched_cellpose_core_use_gpu,
        patched_segment_ROI,
    )

    monkeypatch.setattr(
        "fractal_tasks_core.tasks.cellpose_segmentation.cellpose.core.use_gpu",
        patched_cellpose_core_use_gpu,
    )

    monkeypatch.setattr(
        "fractal_tasks_core.tasks.cellpose_segmentation.segment_ROI",
        patched_segment_ROI,
    )

    # Per-FOV labeling
    for component in metadata["image"]:
        cellpose_segmentation(
            input_paths=[str(root_path)],
            output_path=str(root_path),
            input_ROI_table="grid_ROI_table",
            metadata=metadata,
            component=component,
            channel=Channel(wavelength_id="Channel 0"),
            level=0,
            relabeling=True,
            diameter_level0=80.0,
            augment=True,
            net_avg=True,
            min_size=30,
        )

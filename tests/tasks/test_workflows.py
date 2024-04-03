"""
Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
University of Zurich

Original authors:
Marco Franzon <marco.franzon@exact-lab.it>
Tommaso Comparin <tommaso.comparin@exact-lab.it>

This file is part of Fractal and was originally developed by eXact lab S.r.l.
<exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
Institute for Biomedical Research and Pelkmans Lab from the University of
Zurich.
"""
import logging
import shutil
from pathlib import Path
from typing import Any

import pytest
import zarr
from devtools import debug

from ._validation import check_file_number
from ._validation import validate_schema
from fractal_tasks_core.tasks.copy_ome_zarr import copy_ome_zarr
from fractal_tasks_core.tasks.create_cellvoyager_ome_zarr_compute import (
    create_cellvoyager_ome_zarr_compute,
)
from fractal_tasks_core.tasks.create_cellvoyager_ome_zarr_init import (
    create_cellvoyager_ome_zarr_init,
)
from fractal_tasks_core.tasks.illumination_correction import (
    illumination_correction,
)
from fractal_tasks_core.tasks.maximum_intensity_projection import (
    maximum_intensity_projection,
)  # noqa
from fractal_tasks_core.zarr_utils import OverwriteNotAllowedError


allowed_channels = [
    {
        "label": "DAPI",
        "wavelength_id": "A01_C01",
        "color": "00FFFF",
        "window": {"start": 0, "end": 700},
    },
    {
        "wavelength_id": "A01_C02",
        "label": "nanog",
        "color": "FF00FF",
        "window": {"start": 0, "end": 180},
    },
    {
        "wavelength_id": "A02_C03",
        "label": "Lamin B1",
        "color": "FFFF00",
        "window": {"start": 0, "end": 1500},
    },
]

num_levels = 6
coarsening_xy = 2


@pytest.mark.xfail(reason="This would fail for a dataset with N>1 channels")
def test_create_ome_zarr_fail(tmp_path: Path, zenodo_images: str):

    tmp_allowed_channels = [
        {"label": "repeated label", "wavelength_id": "A01_C01"},
        {"label": "repeated label", "wavelength_id": "A01_C02"},
        {"label": "repeated label", "wavelength_id": "A02_C03"},
    ]

    # Init
    image_dir = zenodo_images
    zarr_dir = str(tmp_path / "tmp_out/")

    # Create zarr structure
    with pytest.raises(ValueError):
        _ = create_cellvoyager_ome_zarr_init(
            zarr_urls=[],
            zarr_dir=zarr_dir,
            image_dirs=[image_dir],
            allowed_channels=tmp_allowed_channels,
            num_levels=num_levels,
            coarsening_xy=coarsening_xy,
            metadata_table_file=None,
        )


def test_create_ome_zarr_no_images(
    tmp_path: Path,
    zenodo_images: str,
    testdata_path: Path,
):
    """
    For invalid image_extension or image_glob_patterns arguments,
    create_ome_zarr must fail.
    """
    with pytest.raises(ValueError):
        create_cellvoyager_ome_zarr_init(
            zarr_urls=[],
            zarr_dir=str(tmp_path / "output"),
            image_dirs=[zenodo_images],
            allowed_channels=allowed_channels,
            num_levels=num_levels,
            coarsening_xy=coarsening_xy,
            metadata_table_file=None,
            image_extension="xyz",
        )
    with pytest.raises(ValueError):
        create_cellvoyager_ome_zarr_init(
            zarr_urls=[],
            zarr_dir=str(tmp_path / "output"),
            image_dirs=[zenodo_images],
            allowed_channels=allowed_channels,
            num_levels=num_levels,
            coarsening_xy=coarsening_xy,
            metadata_table_file=None,
            image_extension="png",
            image_glob_patterns=["*asdasd*"],
        )


metadata_inputs = ["use_mrf_mlf_files", "use_existing_csv_files"]


@pytest.mark.parametrize("metadata_input", metadata_inputs)
def test_yokogawa_to_ome_zarr(
    tmp_path: Path,
    zenodo_images: str,
    testdata_path: Path,
    metadata_input: str,
):

    # Select the kind of metadata_table_file input
    if metadata_input == "use_mrf_mlf_files":
        metadata_table_file = None
    if metadata_input == "use_existing_csv_files":
        testdata_str = testdata_path.as_posix()
        metadata_table_file = (
            f"{testdata_str}/metadata_files/"
            + "corrected_site_metadata_tiny_test.csv"
        )
    debug(metadata_table_file)

    # Init
    img_path = Path(zenodo_images)
    output_path = tmp_path / "output"

    # Create zarr structure
    parallelization_list = create_cellvoyager_ome_zarr_init(
        zarr_urls=[],
        zarr_dir=str(output_path),
        image_dirs=[str(img_path)],
        allowed_channels=allowed_channels,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        metadata_table_file=metadata_table_file,
        image_extension="png",
    )
    debug(parallelization_list)

    # Re-run (with overwrite=False) and fail
    with pytest.raises(OverwriteNotAllowedError):
        create_cellvoyager_ome_zarr_init(
            zarr_urls=[],
            zarr_dir=str(output_path),
            image_dirs=[str(img_path)],
            allowed_channels=allowed_channels,
            num_levels=num_levels,
            coarsening_xy=coarsening_xy,
            metadata_table_file=metadata_table_file,
            image_extension="png",
            overwrite=False,
        )

    # Re-run (with overwrite=True)
    parallelization_list = create_cellvoyager_ome_zarr_init(
        zarr_urls=[],
        zarr_dir=str(output_path),
        image_dirs=[str(img_path)],
        allowed_channels=allowed_channels,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        metadata_table_file=metadata_table_file,
        image_extension="png",
        overwrite=True,
    )
    debug(parallelization_list)

    image_list_updates = []
    # Yokogawa to zarr
    for image in parallelization_list:
        image_list_updates += create_cellvoyager_ome_zarr_compute(
            zarr_url=image["zarr_url"],
            init_args=image["init_args"],
        )["image_list_updates"]
    debug(image_list_updates)

    # Validate image_list_updates contents
    expected_image_list_update = {
        "zarr_url": (
            f"{output_path}/20200812-CardiomyocyteDifferentiation14"
            "-Cycle1.zarr/B/03/0/"
        ),
        "attributes": {
            "plate": "20200812-CardiomyocyteDifferentiation14-Cycle1.zarr",
            "well": "B03",
        },
        "types": {
            "is_3D": True,
        },
    }

    assert image_list_updates[0] == expected_image_list_update

    # Re-run (with overwrite=True)
    for image in parallelization_list:
        create_cellvoyager_ome_zarr_compute(
            zarr_url=image["zarr_url"],
            init_args=image["init_args"],
            overwrite=True,
        )

    # Re-run (with overwrite=False)
    for image in parallelization_list:
        with pytest.raises(OverwriteNotAllowedError):
            create_cellvoyager_ome_zarr_compute(
                zarr_url=image["zarr_url"],
                init_args=image["init_args"],
                overwrite=False,
            )

    # OME-NGFF JSON validation
    image_zarr = Path(parallelization_list[0]["zarr_url"])
    well_zarr = image_zarr.parent
    plate_zarr = image_zarr.parents[2]
    validate_schema(path=str(image_zarr), type="image")
    validate_schema(path=str(well_zarr), type="well")
    validate_schema(path=str(plate_zarr), type="plate")

    check_file_number(zarr_path=image_zarr)

    # Test presence and attributes of FOV/well ROI tables
    for table_name in ["FOV_ROI_table", "well_ROI_table"]:
        table_attrs = zarr.open_group(
            image_zarr / f"tables/{table_name}", mode="r"
        ).attrs.asdict()
        assert table_attrs["type"] == "roi_table"
        assert table_attrs["fractal_table_version"] == "1"


def test_MIP(
    tmp_path: Path,
    zenodo_zarr: list[str],
    zenodo_zarr_metadata: list[dict[str, Any]],
):

    # Init
    zarr_path = tmp_path / "tmp_out/"
    zarr_path_mip = tmp_path / "tmp_out_mip/"

    # Load zarr array from zenodo
    zenodo_zarr_3D, zenodo_zarr_2D = zenodo_zarr[:]
    metadata_3D, metadata_2D = zenodo_zarr_metadata[:]
    shutil.copytree(zenodo_zarr_3D, str(zarr_path / Path(zenodo_zarr_3D).name))
    metadata = metadata_3D.copy()

    # Replicate
    _original_metadata = metadata.copy()
    metadata_update = copy_ome_zarr(
        input_paths=[str(zarr_path)],
        output_path=str(zarr_path_mip),
        metadata=metadata,
        project_to_2D=True,
        suffix="mip",
    )
    metadata.update(metadata_update)
    debug(metadata)

    # Run again, with overwrite=True
    metadata_update_second_try = copy_ome_zarr(
        input_paths=[str(zarr_path)],
        output_path=str(zarr_path_mip),
        metadata=_original_metadata,
        project_to_2D=True,
        suffix="mip",
        overwrite=True,
    )
    assert metadata_update_second_try == metadata_update

    # Run again, with overwrite=False
    with pytest.raises(OverwriteNotAllowedError):
        copy_ome_zarr(
            input_paths=[str(zarr_path)],
            output_path=str(zarr_path_mip),
            metadata=_original_metadata,
            project_to_2D=True,
            suffix="mip",
            overwrite=False,
        )

    # MIP
    for component in metadata["image"]:
        maximum_intensity_projection(
            input_paths=[str(zarr_path_mip)],
            output_path=str(zarr_path_mip),
            metadata=metadata,
            component=component,
        )

    # Re-run with overwrite=True
    for component in metadata["image"]:
        maximum_intensity_projection(
            input_paths=[str(zarr_path_mip)],
            output_path=str(zarr_path_mip),
            metadata=metadata,
            component=component,
            overwrite=True,
        )

    # Re-run with overwrite=False
    with pytest.raises(OverwriteNotAllowedError):
        for component in metadata["image"]:
            maximum_intensity_projection(
                input_paths=[str(zarr_path_mip)],
                output_path=str(zarr_path_mip),
                metadata=metadata,
                component=component,
                overwrite=False,
            )

    # OME-NGFF JSON validation
    image_zarr = Path(zarr_path_mip / metadata["image"][0])
    debug(image_zarr)
    well_zarr = image_zarr.parent
    plate_zarr = image_zarr.parents[2]
    validate_schema(path=str(image_zarr), type="image")
    validate_schema(path=str(well_zarr), type="well")
    validate_schema(path=str(plate_zarr), type="plate")

    # Test presence and attributes of FOV/well ROI tables
    for table_name in ["FOV_ROI_table", "well_ROI_table"]:
        table_attrs = zarr.open_group(
            image_zarr / f"tables/{table_name}", mode="r"
        ).attrs.asdict()
        assert table_attrs["type"] == "roi_table"
        assert table_attrs["fractal_table_version"] == "1"


# def test_MIP_subset_of_images(
#     tmp_path: Path,
#     zenodo_images: str,
# ):
#     """
#     Run a full image-parsing + MIP workflow on a subset of the images (i.e. a
#     single field of view).
#     """

#     # Init
#     zarr_dir = tmp_path / "tmp_out/"
#     zarr_dir_mip = tmp_path / "tmp_out_mip/"

#     # Create zarr structure
#     parallelization_list = create_cellvoyager_ome_zarr_init(
#         zarr_urls=[],
#         zarr_dir=str(zarr_dir),
#         image_dirs=[zenodo_images],
#         allowed_channels=allowed_channels,
#         num_levels=num_levels,
#         coarsening_xy=coarsening_xy,
#         metadata_table_file=None,
#         image_extension="png",
#         image_glob_patterns=["*F001*"],
#     )
#     debug(parallelization_list)

#     # Yokogawa to zarr
#     image_list_updates = []
#     # Yokogawa to zarr
#     for image in parallelization_list:
#         image_list_updates += create_cellvoyager_ome_zarr_compute(
#                 zarr_url=image["zarr_url"],
#                 init_args=image["init_args"],
#             )["image_list_updates"]
#     debug(image_list_updates)

#     # Replicate
#     metadata_update = copy_ome_zarr(
#         input_paths=[str(zarr_dir)],
#         output_path=str(zarr_dir_mip),
#         metadata=metadata,
#         project_to_2D=True,
#         suffix="mip",
#     )
#     metadata.update(metadata_update)
#     debug(metadata)

#     # MIP
#     for component in metadata["image"]:
#         maximum_intensity_projection(
#             input_paths=[str(zarr_path_mip)],
#             output_path=str(zarr_path_mip),
#             metadata=metadata,
#             component=component,
#         )

#     # OME-NGFF JSON validation
#     image_zarr = Path(zarr_path_mip / metadata["image"][0])
#     debug(image_zarr)
#     well_zarr = image_zarr.parent
#     plate_zarr = image_zarr.parents[2]
#     validate_schema(path=str(image_zarr), type="image")
#     validate_schema(path=str(well_zarr), type="well")
#     validate_schema(path=str(plate_zarr), type="plate")


def test_illumination_correction(
    tmp_path: Path,
    testdata_path: Path,
    zenodo_images: str,
    caplog: pytest.LogCaptureFixture,
):

    # Setup caplog fixture, see
    # https://docs.pytest.org/en/stable/how-to/logging.html#caplog-fixture
    caplog.set_level(logging.INFO)

    # Init
    img_path = Path(zenodo_images)
    zarr_dir = tmp_path / "tmp_out"

    testdata_str = testdata_path.as_posix()
    illum_params = {"A01_C01": "illum_corr_matrix.png"}
    illumination_profiles_folder = f"{testdata_str}/illumination_correction/"

    # Create zarr structure
    parallelization_list = create_cellvoyager_ome_zarr_init(
        zarr_urls=[],
        zarr_dir=str(zarr_dir),
        image_dirs=[str(img_path)],
        image_extension="png",
        allowed_channels=allowed_channels,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        metadata_table_file=None,
    )
    print(caplog.text)
    caplog.clear()

    # Yokogawa to zarr
    for image in parallelization_list:
        create_cellvoyager_ome_zarr_compute(
            zarr_url=image["zarr_url"],
            init_args=image["init_args"],
        )
    print(caplog.text)
    caplog.clear()

    # Illumination correction
    for image in parallelization_list:
        illumination_correction(
            zarr_url=image["zarr_url"],
            overwrite_input=True,
            illumination_profiles_folder=illumination_profiles_folder,
            dict_corr=illum_params,
        )
    print(caplog.text)
    caplog.clear()

    # OME-NGFF JSON validation
    image_zarr = Path(parallelization_list[0]["zarr_url"])
    well_zarr = image_zarr.parent
    plate_zarr = image_zarr.parents[2]
    validate_schema(path=str(image_zarr), type="image")
    validate_schema(path=str(well_zarr), type="well")
    validate_schema(path=str(plate_zarr), type="plate")

    check_file_number(zarr_path=image_zarr)


def test_copy_ome_zarr_twice(
    tmp_path: Path,
    zenodo_zarr: list[str],
    zenodo_zarr_metadata: list[dict[str, Any]],
):
    """
    See
    https://github.com/fractal-analytics-platform/fractal-tasks-core/issues/510
    """

    # Init and load zarr array from zenodo
    zarr_path = tmp_path / "data"
    zenodo_zarr_3D, zenodo_zarr_2D = zenodo_zarr[:]
    metadata_3D, metadata_2D = zenodo_zarr_metadata[:]
    shutil.copytree(zenodo_zarr_3D, str(zarr_path / Path(zenodo_zarr_3D).name))
    metadata = metadata_3D.copy()

    # Run first time
    _original_metadata = metadata.copy()
    metadata_update = copy_ome_zarr(
        input_paths=[str(zarr_path)],
        output_path=str(zarr_path),
        metadata=_original_metadata,
        project_to_2D=True,
        suffix="mip",
    )
    metadata.update(metadata_update)
    debug(metadata)
    assert len(metadata["copy_ome_zarr"]["sources"].keys()) == 1
    assert "plate_mip_mip" not in metadata["copy_ome_zarr"]["sources"].keys()

    # Run second time, with overwrite=True
    metadata_update = copy_ome_zarr(
        input_paths=[str(zarr_path)],
        output_path=str(zarr_path),
        metadata=_original_metadata,
        project_to_2D=True,
        suffix="mip",
        overwrite=True,
    )
    debug(metadata_update)
    metadata.update(metadata_update)
    assert (
        "plate_mip_mip"
        not in metadata_update["copy_ome_zarr"]["sources"].keys()
    )
    assert len(metadata_update["copy_ome_zarr"]["sources"].keys()) == 1

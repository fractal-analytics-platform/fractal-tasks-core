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

import anndata as ad
import dask.array as da
import pytest
import zarr
from devtools import debug
from ngio.utils import NgioFileExistsError

from ._validation import check_file_number
from ._validation import validate_schema
from fractal_tasks_core.tasks.cellvoyager_to_ome_zarr_compute import (
    cellvoyager_to_ome_zarr_compute,
)
from fractal_tasks_core.tasks.cellvoyager_to_ome_zarr_init import (
    cellvoyager_to_ome_zarr_init,
)
from fractal_tasks_core.tasks.copy_ome_zarr_hcs_plate import (
    copy_ome_zarr_hcs_plate,
)
from fractal_tasks_core.tasks.illumination_correction import (
    illumination_correction,
)
from fractal_tasks_core.tasks.projection import (
    projection,
)
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
        _ = cellvoyager_to_ome_zarr_init(
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
    For invalid image_extension or include_glob_patterns arguments,
    create_ome_zarr must fail.
    """
    with pytest.raises(ValueError):
        cellvoyager_to_ome_zarr_init(
            zarr_dir=str(tmp_path / "output"),
            image_dirs=[zenodo_images],
            allowed_channels=allowed_channels,
            num_levels=num_levels,
            coarsening_xy=coarsening_xy,
            metadata_table_file=None,
            image_extension="xyz",
        )
    with pytest.raises(ValueError):
        cellvoyager_to_ome_zarr_init(
            zarr_dir=str(tmp_path / "output"),
            image_dirs=[zenodo_images],
            allowed_channels=allowed_channels,
            num_levels=num_levels,
            coarsening_xy=coarsening_xy,
            metadata_table_file=None,
            image_extension="png",
            include_glob_patterns=["*asdasd*"],
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
    parallelization_list = cellvoyager_to_ome_zarr_init(
        zarr_dir=str(output_path),
        image_dirs=[str(img_path)],
        allowed_channels=allowed_channels,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        metadata_table_file=metadata_table_file,
        image_extension="png",
    )["parallelization_list"]
    debug(parallelization_list)

    image_list_updates = []
    # Yokogawa to zarr
    for image in parallelization_list:
        image_list_updates += cellvoyager_to_ome_zarr_compute(
            zarr_url=image["zarr_url"],
            init_args=image["init_args"],
        )["image_list_updates"]
    debug(image_list_updates)

    # Validate image_list_updates contents
    expected_image_list_update = {
        "zarr_url": (
            f"{output_path}/20200812-CardiomyocyteDifferentiation14"
            "-Cycle1.zarr/B/03/0"
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

    # Re-run (with overwrite=True for the init task)
    parallelization_list = cellvoyager_to_ome_zarr_init(
        zarr_dir=str(output_path),
        image_dirs=[str(img_path)],
        allowed_channels=allowed_channels,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        metadata_table_file=metadata_table_file,
        image_extension="png",
        overwrite=True,
    )["parallelization_list"]

    for image in parallelization_list:
        cellvoyager_to_ome_zarr_compute(
            zarr_url=image["zarr_url"],
            init_args=image["init_args"],
        )

    # Re-run (with overwrite=False for the init task) and fail
    with pytest.raises(OverwriteNotAllowedError):
        cellvoyager_to_ome_zarr_init(
            zarr_dir=str(output_path),
            image_dirs=[str(img_path)],
            allowed_channels=allowed_channels,
            num_levels=num_levels,
            coarsening_xy=coarsening_xy,
            metadata_table_file=metadata_table_file,
            image_extension="png",
            overwrite=False,
        )


def test_2D_cellvoyager_to_ome_zarr(
    tmp_path: Path,
    zenodo_images: str,
):
    # Init
    output_path = tmp_path / "output"

    # Create zarr structure
    parallelization_list = cellvoyager_to_ome_zarr_init(
        zarr_dir=str(output_path),
        image_dirs=[zenodo_images],
        allowed_channels=allowed_channels,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        include_glob_patterns=["*Z01*"],
        image_extension="png",
    )["parallelization_list"]
    debug(parallelization_list)

    image_list_updates = []
    # Yokogawa to zarr
    for image in parallelization_list:
        image_list_updates += cellvoyager_to_ome_zarr_compute(
            zarr_url=image["zarr_url"],
            init_args=image["init_args"],
        )["image_list_updates"]
    debug(image_list_updates)

    # Validate image_list_updates contents
    expected_image_list_update = {
        "zarr_url": (
            f"{output_path}/20200812-CardiomyocyteDifferentiation14"
            "-Cycle1.zarr/B/03/0"
        ),
        "attributes": {
            "plate": "20200812-CardiomyocyteDifferentiation14-Cycle1.zarr",
            "well": "B03",
        },
        "types": {
            "is_3D": False,
        },
    }

    assert image_list_updates[0] == expected_image_list_update

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
):

    # Init
    zarr_path = tmp_path / "tmp_out"
    debug(zarr_path)

    # Load zarr array from zenodo
    zenodo_zarr_3D, zenodo_zarr_2D = zenodo_zarr[:]
    shutil.copytree(zenodo_zarr_3D, str(zarr_path / Path(zenodo_zarr_3D).name))

    zarr_urls = []
    zarr_urls = [Path(zarr_path, "plate.zarr/B/03/0").as_posix()]

    parallelization_list = copy_ome_zarr_hcs_plate(
        zarr_urls=zarr_urls,
        zarr_dir=str(zarr_path),
        overwrite_images=True,
        re_initialize_plate=True,
    )["parallelization_list"]
    debug(parallelization_list)

    # Run again, with overwrite=True
    parallelization_list_2 = copy_ome_zarr_hcs_plate(
        zarr_urls=zarr_urls,
        zarr_dir=str(zarr_path),
        overwrite_images=True,
        re_initialize_plate=True,
    )["parallelization_list"]
    assert parallelization_list_2 == parallelization_list

    # MIP
    image_list_updates = []
    for image in parallelization_list:
        image_list_updates += projection(
            zarr_url=image["zarr_url"],
            init_args=image["init_args"],
        )["image_list_updates"]

    debug(image_list_updates[0])
    expected_image_list_updates = {
        "zarr_url": (parallelization_list[0]["zarr_url"]),
        "origin": f"{zarr_path}/plate.zarr/B/03/0",
        "types": {
            "is_3D": False,
        },
        "attributes": {"plate": "plate_mip.zarr"},
    }
    debug(expected_image_list_updates)
    assert image_list_updates[0] == expected_image_list_updates

    # Re-run with overwrite=True
    for image in parallelization_list:
        projection(
            zarr_url=image["zarr_url"],
            init_args=image["init_args"],
        )

    with pytest.raises(NgioFileExistsError):
        for image in parallelization_list:
            image["init_args"]["overwrite"] = False
            projection(
                zarr_url=image["zarr_url"],
                init_args=image["init_args"],
            )
    # OME-NGFF JSON validation
    image_zarr = Path(parallelization_list[0]["zarr_url"])
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

    # Check correct zarr metadata for row folder (issue #780): Checks that
    # the one well expected to be in the Zarr plate is discoverable by the
    # Zarr API
    plate_zarr_group = zarr.open(plate_zarr)
    assert len(plate_zarr_group) == 1
    row_zarr_group = zarr.open(plate_zarr / "B")
    assert len(row_zarr_group) == 1


@pytest.mark.parametrize("method", ["mip", "minip", "meanip", "sumip"])
def test_projection_methods(
    tmp_path: Path,
    zenodo_zarr: list[str],
    method: str,
):
    # Init
    zarr_path = tmp_path / "tmp_out"
    debug(zarr_path)

    # Load zarr array from zenodo
    zenodo_zarr_3D, zenodo_zarr_2D = zenodo_zarr[:]
    shutil.copytree(zenodo_zarr_3D, str(zarr_path / Path(zenodo_zarr_3D).name))

    zarr_urls = []
    zarr_urls = [Path(zarr_path, "plate.zarr/B/03/0").as_posix()]

    parallelization_list = copy_ome_zarr_hcs_plate(
        zarr_urls=zarr_urls,
        zarr_dir=str(zarr_path),
        method=method,
        overwrite_images=True,
        re_initialize_plate=True,
    )["parallelization_list"]

    # Check that method is correctly in parallelization list & in suffix
    expected_parallelization_list = [
        {
            "zarr_url": Path(
                zarr_path, f"plate_{method}.zarr/B/03/0"
            ).as_posix(),
            "init_args": {
                "origin_url": zarr_urls[0],
                "method": method,
                "overwrite": True,
                "new_plate_name": f"plate_{method}.zarr",
            },
        }
    ]
    assert parallelization_list == expected_parallelization_list

    # Run projection
    image_list_updates = []
    for image in parallelization_list:
        image_list_updates += projection(
            zarr_url=image["zarr_url"],
            init_args=image["init_args"],
        )["image_list_updates"]

    # Test the value of a pre-defined pixel to see if the projection worked
    img = da.from_zarr(f"{image_list_updates[0]['zarr_url']}/0")
    expected_values = {
        "mip": 493,
        "minip": 332,
        "meanip": 412,
        "sumip": 825,
    }

    debug(f"{method}:{img[0, 0, 952, 735].compute()}")
    debug(img.dtype)
    assert img[0, 0, 952, 735].compute() == expected_values[method]


def test_MIP_subset_of_images(
    tmp_path: Path,
    zenodo_images: str,
):
    """
    Run a full image-parsing + MIP workflow on a subset of the images (i.e. a
    single field of view).
    """

    # Init
    zarr_dir = tmp_path / "tmp_out/"

    # Create zarr structure
    parallelization_list = cellvoyager_to_ome_zarr_init(
        zarr_dir=str(zarr_dir),
        image_dirs=[zenodo_images],
        allowed_channels=allowed_channels,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        metadata_table_file=None,
        image_extension="png",
        include_glob_patterns=["*F001*"],
    )["parallelization_list"]
    debug(parallelization_list)

    # Yokogawa to zarr
    image_list_updates = []
    # Yokogawa to zarr
    for image in parallelization_list:
        image_list_updates += cellvoyager_to_ome_zarr_compute(
            zarr_url=image["zarr_url"],
            init_args=image["init_args"],
        )["image_list_updates"]
    debug(image_list_updates)

    zarr_urls = [a["zarr_url"] for a in image_list_updates]
    debug(zarr_urls)
    # Replicate
    parallelization_list = copy_ome_zarr_hcs_plate(
        zarr_urls=zarr_urls,
        zarr_dir=str(zarr_dir),
        overwrite_images=True,
        re_initialize_plate=True,
    )["parallelization_list"]
    debug(parallelization_list)

    # MIP
    for image in parallelization_list:
        projection(
            zarr_url=image["zarr_url"],
            init_args=image["init_args"],
        )

    # OME-NGFF JSON validation
    image_zarr = Path(parallelization_list[0]["zarr_url"])
    debug(image_zarr)
    well_zarr = image_zarr.parent
    plate_zarr = image_zarr.parents[2]
    validate_schema(path=str(image_zarr), type="image")
    validate_schema(path=str(well_zarr), type="well")
    validate_schema(path=str(plate_zarr), type="plate")


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
    parallelization_list = cellvoyager_to_ome_zarr_init(
        zarr_dir=str(zarr_dir),
        image_dirs=[str(img_path)],
        image_extension="png",
        allowed_channels=allowed_channels,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        metadata_table_file=None,
    )["parallelization_list"]
    print(caplog.text)
    caplog.clear()

    # Yokogawa to zarr
    for image in parallelization_list:
        cellvoyager_to_ome_zarr_compute(
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
            illumination_profiles=illum_params,
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


def test_yokogawa_to_ome_zarr_multiplate(
    tmp_path: Path,
    zenodo_images_multiplex: str,
):
    img_path_1, img_path_2 = zenodo_images_multiplex
    output_path = tmp_path / "output"

    # Create zarr structure
    parallelization_list = cellvoyager_to_ome_zarr_init(
        zarr_dir=str(output_path),
        image_dirs=[img_path_1, img_path_2],
        allowed_channels=allowed_channels,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        image_extension="png",
        overwrite=False,
    )["parallelization_list"]
    debug(parallelization_list)

    image_list_updates = []
    # Yokogawa to zarr
    for image in parallelization_list:
        image_list_updates += cellvoyager_to_ome_zarr_compute(
            zarr_url=image["zarr_url"],
            init_args=image["init_args"],
        )["image_list_updates"]
    debug(image_list_updates)

    # Validate image_list_updates contents
    expected_image_list_update = [
        {
            "zarr_url": (
                f"{output_path}/20200812-CardiomyocyteDifferentiation14"
                "-Cycle1.zarr/B/03/0"
            ),
            "attributes": {
                "plate": "20200812-CardiomyocyteDifferentiation14-Cycle1.zarr",
                "well": "B03",
            },
            "types": {
                "is_3D": True,
            },
        },
        {
            "zarr_url": (
                f"{output_path}/20200812-CardiomyocyteDifferentiation14"
                "-Cycle1_1.zarr/B/03/0"
            ),
            "attributes": {
                "plate": "20200812-CardiomyocyteDifferentiation14-Cycle1_1.zarr",  # noqa
                "well": "B03",
            },
            "types": {
                "is_3D": True,
            },
        },
    ]
    debug(image_list_updates)
    debug(expected_image_list_update)

    assert image_list_updates == expected_image_list_update


def test_cellvoyager_converter_exclusion_patterns(
    tmp_path: Path,
    zenodo_images: str,
):
    # Init
    output_path = tmp_path / "output"
    debug(output_path)

    # Create zarr structure
    parallelization_list = cellvoyager_to_ome_zarr_init(
        zarr_dir=str(output_path),
        image_dirs=[zenodo_images],
        allowed_channels=allowed_channels,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        exclude_glob_patterns=["*F002*"],
        image_extension="png",
    )["parallelization_list"]
    debug(parallelization_list)

    image_list_updates = []
    # Yokogawa to zarr
    for image in parallelization_list:
        image_list_updates += cellvoyager_to_ome_zarr_compute(
            zarr_url=image["zarr_url"],
            init_args=image["init_args"],
        )["image_list_updates"]
    debug(image_list_updates)

    # # OME-NGFF JSON validation
    image_zarr = Path(parallelization_list[0]["zarr_url"])
    validate_schema(path=str(image_zarr), type="image")

    # Check that FOV ROI table now only has a single FOV
    fov_roi_table = ad.read_zarr(image_zarr / "tables/FOV_ROI_table")
    assert len(fov_roi_table) == 1

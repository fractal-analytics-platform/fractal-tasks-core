import glob
import os
import shutil
from pathlib import Path

import anndata as ad
import dask.array as da
import numpy as np
import pandas as pd
import pytest
from devtools import debug
from PIL import Image

from fractal_tasks_core.lib_regions_of_interest import (
    convert_indices_to_regions,
)
from fractal_tasks_core.lib_regions_of_interest import (
    convert_ROI_table_to_indices,
)
from fractal_tasks_core.lib_regions_of_interest import load_region
from fractal_tasks_core.lib_zattrs_utils import extract_zyx_pixel_sizes
from fractal_tasks_core.tasks.apply_registration_to_image import (
    apply_registration_to_image,
)
from fractal_tasks_core.tasks.apply_registration_to_ROI_table import (
    apply_registration_to_ROI_table,
)
from fractal_tasks_core.tasks.calculate_2D_registration_image_based import (
    calculate_2D_registration_image_based,
)
from fractal_tasks_core.tasks.copy_ome_zarr import (
    copy_ome_zarr,
)
from fractal_tasks_core.tasks.create_ome_zarr_multiplex import (
    create_ome_zarr_multiplex,
)
from fractal_tasks_core.tasks.maximum_intensity_projection import (
    maximum_intensity_projection,
)
from fractal_tasks_core.tasks.yokogawa_to_ome_zarr import yokogawa_to_ome_zarr


single_cycle_allowed_channels_no_label = [
    {
        "wavelength_id": "A01_C01",
        "color": "00FFFF",
        "window": {"start": 0, "end": 700},
    }
]

allowed_channels = {
    "0": single_cycle_allowed_channels_no_label,
    "1": single_cycle_allowed_channels_no_label,
}


def _shift_image(
    img_path: str,
    shift_x_pxl: int = 200,
    shift_y_pxl: int = 50,
) -> None:
    """
    Load an image, apply a XY shift, replace the remaining stripes with noise.
    """

    # Open old image as array
    old_img = Image.open(img_path)
    new_array = np.asarray(old_img).copy()

    # Shift image by (shift_y_pxl, shift_x_pxl)
    new_array[:, :-shift_x_pxl] = new_array[:, shift_x_pxl:]
    new_array[:-shift_y_pxl] = new_array[shift_y_pxl:, :]

    # Replace offset stripes with random values
    new_array[:, -shift_x_pxl:] = np.random.randint(
        0, 1000, size=new_array[:, -shift_x_pxl:].shape
    )
    new_array[-shift_y_pxl:, :] = np.random.randint(
        0, 1000, size=new_array[-shift_y_pxl:, :].shape
    )

    # Save new image
    new_img = Image.fromarray(new_array, mode="I")
    new_img.save(img_path, mode="png")


@pytest.fixture(scope="session")
def zenodo_images_multiplex_shifted(
    zenodo_images_multiplex: list[str],
    testdata_path: Path,
) -> list[str]:
    """
    Return a list of strings, like
    ```
    [
        "/some/path/fake_multiplex_shifted/cycle1",
        "/some/path/fake_multiplex_shifted/cycle2"
    ]
    ```
    """
    # Define old and new folders
    old_folder = str(testdata_path / "fake_multiplex")
    new_folder = old_folder.replace("fake_multiplex", "fake_multiplex_shifted")

    # Define output folders (one per multiplexing cycle)
    cycle_folders = [f"{new_folder}/cycle{ind}" for ind in (1, 2)]

    if os.path.isdir(new_folder):
        # If the shifted-images folder already exists, return immediately
        print(f"{new_folder} already exists")
        return cycle_folders
    else:
        # Copy the fake_multiplex folder into a new one
        shutil.copytree(old_folder, new_folder)
        # Loop over images of cycle2 and apply a shift
        for img_path in glob.glob(f"{cycle_folders[1]}/2020*.png"):
            print(f"Now shifting {img_path}")
            _shift_image(str(img_path))
        return cycle_folders


expected_shift = {
    "FOV_ROI_table": [0.0, 7.8, 32.5],
}
registered_columns = [
    "x_micrometer",
    "y_micrometer",
    "z_micrometer",
    "len_x_micrometer",
    "len_y_micrometer",
    "len_z_micrometer",
]

expected_registered_table = {
    "FOV_ROI_table": {
        "20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr/B/03/0": pd.DataFrame(  # noqa: E501
            {
                "x_micrometer": [32.5, 448.5],
                "y_micrometer": [7.8, 8.45],
                "z_micrometer": [0.0, 0.0],
                "len_x_micrometer": [383.5, 383.5],
                "len_y_micrometer": [343.200012, 342.549988],
                "len_z_micrometer": [1.0, 1.0],
            },
            index=["FOV_1", "FOV_2"],
        ).rename_axis(
            "FieldIndex"
        ),
        "20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr/B/03/1": pd.DataFrame(  # noqa: E501
            {
                "x_micrometer": [0.0, 416.0],
                "y_micrometer": [0, 0],
                "z_micrometer": [0.0, 0.0],
                "len_x_micrometer": [383.5, 383.5],
                "len_y_micrometer": [343.200012, 342.549988],
                "len_z_micrometer": [1.0, 1.0],
            },
            index=["FOV_1", "FOV_2"],
        ).rename_axis(
            "FieldIndex"
        ),
    }
}


def test_multiplexing_registration(
    zenodo_images_multiplex_shifted: list[str],
    tmp_path,
    roi_table="FOV_ROI_table",  # Given the test data, only implemented per FOV
):
    zarr_path = tmp_path / "registration_output/"
    zarr_path_mip = tmp_path / "registration_output_mip/"
    metadata = {}
    # Create the multiplexed OME-Zarr
    metadata_update = create_ome_zarr_multiplex(
        input_paths=zenodo_images_multiplex_shifted,
        output_path=str(zarr_path),
        metadata={},
        image_extension="png",
        allowed_channels=allowed_channels,
    )
    metadata.update(metadata_update)
    debug(metadata)

    for component in metadata["image"]:
        yokogawa_to_ome_zarr(
            input_paths=[str(zarr_path)],
            output_path=str(zarr_path),
            metadata=metadata,
            component=component,
        )
    debug(metadata)

    # Replicate
    metadata_update = copy_ome_zarr(
        input_paths=[str(zarr_path)],
        output_path=str(zarr_path_mip),
        metadata=metadata,
        project_to_2D=True,
        suffix="mip",
    )
    metadata.update(metadata_update)
    debug(metadata)

    # MIP
    for component in metadata["image"]:
        maximum_intensity_projection(
            input_paths=[str(zarr_path_mip)],
            output_path=str(zarr_path_mip),
            metadata=metadata,
            component=component,
        )

    # Calculate registration
    for component in metadata["image"]:
        calculate_2D_registration_image_based(
            input_paths=[str(zarr_path_mip)],
            output_path=str(zarr_path_mip),
            metadata=metadata,
            component=component,
            wavelength_id="A01_C01",
            roi_table=roi_table,
        )

    # Check the table for the second component (the image of the second cycle)
    component = metadata["image"][1]
    curr_table = ad.read_zarr(
        f"{zarr_path_mip / component}/tables/{roi_table}"
    )
    assert curr_table.shape == (2, 11)
    np.testing.assert_almost_equal(
        curr_table.X[0, 8:11], np.array(expected_shift[roi_table]), decimal=5
    )
    # Apply registration to ROI table
    for component in metadata["well"]:
        apply_registration_to_ROI_table(
            input_paths=[str(zarr_path_mip)],
            output_path=str(zarr_path_mip),
            metadata=metadata,
            component=component,
            roi_table=roi_table,
        )

    # Validate the aligned tables
    for component in metadata["image"]:
        registered_table = ad.read_zarr(
            f"{zarr_path_mip / component}/tables/registered_{roi_table}"
        )
        pd.testing.assert_frame_equal(
            registered_table.to_df()[registered_columns].astype("float32"),
            expected_registered_table[roi_table][component].astype("float32"),
            check_column_type=False,
        )

    # Apply registration to image
    for component in metadata["image"]:
        apply_registration_to_image(
            input_paths=[str(zarr_path_mip)],
            output_path=str(zarr_path_mip),
            metadata=metadata,
            component=component,
            registered_roi_table="registered_" + roi_table,
        )

    # Load the Zarr image
    # How many padded pixels are expected => number of pixels that are 0
    # a) many when loading the original ROI
    # b) none when loading the registered ROI
    for component in metadata["image"]:
        # Read pixel sizes from zattrs file
        pxl_sizes_zyx = extract_zyx_pixel_sizes(
            f"{str(zarr_path_mip / component)}/.zattrs", level=0
        )

        original_table = ad.read_zarr(
            f"{zarr_path_mip / component}/tables/{roi_table}"
        )
        # Create list of indices for 3D ROIs
        list_indices = convert_ROI_table_to_indices(
            original_table,
            level=0,
            coarsening_xy=2,
            full_res_pxl_sizes_zyx=pxl_sizes_zyx,
            reset_origin=True,
        )
        region = convert_indices_to_regions(list_indices[0])
        data_array = da.from_zarr(f"{zarr_path_mip / component / str(0)}")[0]
        img_array = load_region(
            data_zyx=data_array, region=region, compute=True
        )
        assert np.sum(img_array == 0) == 545280

        registered_table = ad.read_zarr(
            f"{zarr_path_mip / component}/tables/registered_{roi_table}"
        )
        # Create list of indices for 3D ROIs
        list_indices = convert_ROI_table_to_indices(
            registered_table,
            level=0,
            coarsening_xy=2,
            full_res_pxl_sizes_zyx=pxl_sizes_zyx,
            reset_origin=False,  # no reset of origin!
        )
        region = convert_indices_to_regions(list_indices[0])
        data_array = da.from_zarr(f"{zarr_path_mip / component / str(0)}")[0]
        img_array_reg = load_region(
            data_zyx=data_array, region=region, compute=True
        )
        assert np.sum(img_array_reg == 0) == 0

import glob
import json
import os
import shutil
from pathlib import Path

import anndata as ad
import dask.array as da
import imageio
import numpy as np
import pandas as pd
import pytest
from devtools import debug
from pytest import MonkeyPatch

from fractal_tasks_core.ngff.zarr_utils import load_NgffImageMeta
from fractal_tasks_core.roi import (
    convert_indices_to_regions,
)
from fractal_tasks_core.roi import (
    convert_ROI_table_to_indices,
)
from fractal_tasks_core.roi import load_region
from fractal_tasks_core.tasks.cellvoyager_to_ome_zarr_compute import (
    cellvoyager_to_ome_zarr_compute,
)
from fractal_tasks_core.tasks.cellvoyager_to_ome_zarr_init_multiplex import (
    cellvoyager_to_ome_zarr_init_multiplex,
)
from fractal_tasks_core.tasks.copy_ome_zarr_hcs_plate import (
    copy_ome_zarr_hcs_plate,
)
from fractal_tasks_core.tasks.image_based_registration_hcs_init import (
    image_based_registration_hcs_init,
)
from fractal_tasks_core.tasks.io_models import MultiplexingAcquisition
from fractal_tasks_core.tasks.maximum_intensity_projection import (
    maximum_intensity_projection,
)


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
    img = imageio.v2.imread(img_path)

    # Shift image by (shift_y_pxl, shift_x_pxl)
    img[:, :-shift_x_pxl] = img[:, shift_x_pxl:]
    img[:-shift_y_pxl] = img[shift_y_pxl:, :]

    img[:, -shift_x_pxl:] = np.ones(img[:, -shift_x_pxl:].shape) * 110
    img[-shift_y_pxl:, :] = np.ones(img[-shift_y_pxl:, :].shape) * 110

    # Save new image
    imageio.v2.imwrite(
        uri=img_path,
        im=img,
    )


@pytest.fixture(scope="function")
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
                "y_micrometer": [7.8, 7.8],
                "z_micrometer": [0.0, 0.0],
                "len_x_micrometer": [383.5, 383.5],
                "len_y_micrometer": [343.2, 343.2],
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
                "len_y_micrometer": [343.2, 343.2],
                "len_z_micrometer": [1.0, 1.0],
            },
            index=["FOV_1", "FOV_2"],
        ).rename_axis(
            "FieldIndex"
        ),
    }
}


def patched_segment_ROI(
    x, do_3D=True, label_dtype=None, well_id=None, **kwargs
):
    # Expects x to always be a 4D image

    import logging

    logger = logging.getLogger("cellpose_segmentation.py")

    logger.info(f"[{well_id}][patched_segment_ROI] START")
    assert x.ndim == 4
    # Actual labeling: segment_ROI returns a 3D mask with the same shape as x,
    # except for the first dimension
    mask = np.zeros_like(x[0, :, :, :])
    nz, ny, nx = mask.shape
    if do_3D:
        mask[:, 0 : ny // 4, 0 : nx // 4] = 1  # noqa
        mask[:, ny // 4 : ny // 2, 0:nx] = 2  # noqa
    else:
        mask[:, 0 : ny // 4, 0 : nx // 4] = 1  # noqa
        mask[:, ny // 4 : ny // 2, 0:nx] = 2  # noqa

    logger.info(f"[{well_id}][patched_segment_ROI] END")

    return mask.astype(label_dtype)


def test_multiplexing_hcs_init(
    zenodo_images_multiplex_shifted: list[str],
    tmp_path,
    monkeypatch: MonkeyPatch,
    # Given the test data, only implemented per FOV
    roi_table="FOV_ROI_table",
):

    zarr_dir = str(tmp_path / "registration_output/")
    num_levels = 2

    # Init

    acquisitions = {
        "0": MultiplexingAcquisition(
            image_dir=zenodo_images_multiplex_shifted[0],
            allowed_channels=single_cycle_allowed_channels_no_label,
        ),
        "1": MultiplexingAcquisition(
            image_dir=zenodo_images_multiplex_shifted[1],
            allowed_channels=single_cycle_allowed_channels_no_label,
        ),
    }

    # # Create zarr structure
    parallelization_list = cellvoyager_to_ome_zarr_init_multiplex(
        zarr_urls=[],
        zarr_dir=zarr_dir,
        acquisitions=acquisitions,
        num_levels=num_levels,
        coarsening_xy=2,
        image_extension="png",
        metadata_table_files=None,
    )["parallelization_list"]
    debug(parallelization_list)

    # # Convert to OME-Zarr
    image_list_updates = []
    for image in parallelization_list:
        image_list_updates += cellvoyager_to_ome_zarr_compute(
            zarr_url=image["zarr_url"],
            init_args=image["init_args"],
        )["image_list_updates"]
    debug(image_list_updates)

    zarr_urls_3D = [image["zarr_url"] for image in image_list_updates]
    debug(zarr_urls_3D)

    parallelization_list = copy_ome_zarr_hcs_plate(
        zarr_urls=zarr_urls_3D,
        zarr_dir="tmp_out",
        overwrite=True,
    )["parallelization_list"]
    debug(parallelization_list)

    # # MIP
    image_list_updates = []
    for image in parallelization_list:
        image_list_updates += maximum_intensity_projection(
            zarr_url=image["zarr_url"],
            init_args=image["init_args"],
            overwrite=True,
        )["image_list_updates"]

    zarr_urls_2D = [image["zarr_url"] for image in image_list_updates]
    debug(zarr_urls_2D)

    # Test non-available reference cycle:
    with pytest.raises(ValueError):
        image_based_registration_hcs_init(
            zarr_urls=zarr_urls_2D,
            zarr_dir=zarr_dir,
            reference_cycle=2,
        )

    parallelization_list = image_based_registration_hcs_init(
        zarr_urls=zarr_urls_2D,
        zarr_dir=zarr_dir,
    )["parallelization_list"]
    debug(parallelization_list)

    expected_par_list = [
        {
            "zarr_url": zarr_urls_2D[1],
            "init_args": {
                "reference_zarr_url": zarr_urls_2D[0],
            },
        },
    ]
    assert expected_par_list == parallelization_list


# def test_multiplexing_registration(
#     zenodo_images_multiplex_shifted: list[str],
#     tmp_path,
#     monkeypatch: MonkeyPatch,
#     # Given the test data, only implemented per FOV
#     roi_table="FOV_ROI_table",
# ):

#     monkeypatch.setattr(
#         "fractal_tasks_core.tasks.cellpose_segmentation.segment_ROI",
#         patched_segment_ROI,
#     )

#     zarr_dir = tmp_path / "registration_output/"
#     zarr_dir_mip = tmp_path / "registration_output_mip/"
#     metadata = {}
#     # Create the multiplexed OME-Zarr
#     metadata_update = create_ome_zarr_multiplex(
#         input_paths=zenodo_images_multiplex_shifted,
#         output_path=str(zarr_dir),
#         metadata={},
#         image_extension="png",
#         allowed_channels=allowed_channels,
#     )
#     metadata.update(metadata_update)
#     debug(metadata)

#     for component in metadata["image"]:
#         yokogawa_to_ome_zarr(
#             input_paths=[str(zarr_dir)],
#             output_path=str(zarr_dir),
#             metadata=metadata,
#             component=component,
#         )
#     debug(metadata)

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
#             input_paths=[str(zarr_dir_mip)],
#             output_path=str(zarr_dir_mip),
#             metadata=metadata,
#             component=component,
#         )

#     # Cellpose segmentation (so that we test handling of label images)
#     for component in metadata["image"]:
#         zarr_url = str(zarr_dir_mip / component)
#         cellpose_segmentation(
#             zarr_url=zarr_url,
#             level=1,
#             channel=ChannelInputModel(wavelength_id="A01_C01"),
#         )

#     # Calculate registration
#     for component in metadata["image"]:
#         calculate_registration_image_based(
#             input_paths=[str(zarr_dir_mip)],
#             output_path=str(zarr_dir_mip),
#             metadata=metadata,
#             component=component,
#             wavelength_id="A01_C01",
#             roi_table=roi_table,
#         )
#     # Check the table for the second component
#     # (the image of the second cycle)
#     component = metadata["image"][1]
#     curr_table = ad.read_zarr(
#         f"{zarr_dir_mip / component}/tables/{roi_table}"
#     )
#     assert curr_table.shape == (2, 11)
#     np.testing.assert_almost_equal(
#         curr_table.X[0, 8:11], np.array(expected_shift[roi_table]), decimal=5
#     )
#     # Apply registration to ROI table
#     for component in metadata["well"]:
#         apply_registration_to_ROI_tables(
#             input_paths=[str(zarr_dir_mip)],
#             output_path=str(zarr_dir_mip),
#             metadata=metadata,
#             component=component,
#             roi_table=roi_table,
#         )

#     # Validate the aligned tables
#     for component in metadata["image"]:
#         registered_table = ad.read_zarr(
#             f"{zarr_dir_mip / component}/tables/registered_{roi_table}"
#         )
#         pd.testing.assert_frame_equal(
#             registered_table.to_df()[registered_columns].astype("float32"),
#             expected_registered_table[roi_table][component].astype("float32"),
#             check_column_type=False,
#         )

#     # Apply registration to image without overwrite_input and validate the
#     # output
#     zarr_list = []
#     for component in metadata["image"]:
#         zarr_url = str(zarr_dir_mip / component)
#         zarr_list.append(f"{zarr_url}_registered")
#         image_list_update = apply_registration_to_image(
#             zarr_url=zarr_url,
#             registered_roi_table="registered_" + roi_table,
#             overwrite_input=False,
#         )
#         assert image_list_update == dict(
#             image_list_updates=[
#                 dict(zarr_url=f"{zarr_url}_registered", origin=zarr_url)
#             ]
#         )

#     validate_assumptions_after_image_registration(
#         zarr_list=zarr_list,
#         roi_table=roi_table,
#     )

#     # # Apply registration to image with overwrite_input and validate the
#     # # output
#     zarr_list = []
#     for component in metadata["image"]:
#         zarr_url = str(zarr_dir_mip / component)
#         zarr_list.append(zarr_url)
#         image_list_update = apply_registration_to_image(
#             zarr_url=zarr_url,
#             registered_roi_table="registered_" + roi_table,
#             overwrite_input=True,
#         )
#         assert image_list_update == dict(
#             image_list_updates=[dict(zarr_url=zarr_url)]
#         )
#     validate_assumptions_after_image_registration(
#         zarr_list=zarr_list,
#         roi_table=roi_table,
#     )


def validate_assumptions_after_image_registration(
    zarr_list,
    roi_table,
):
    # Load the Zarr image
    # How many padded pixels are expected => number of pixels that are 0
    # a) many when loading the original ROI
    # b) none when loading the registered ROI
    for zarr_url in zarr_list:
        print(zarr_url)
        # Read pixel sizes from zattrs file
        ngff_image_meta = load_NgffImageMeta(zarr_url)
        pxl_sizes_zyx = ngff_image_meta.get_pixel_sizes_zyx(level=0)

        original_table = ad.read_zarr(f"{zarr_url}/tables/{roi_table}")
        # Create list of indices for 3D ROIs
        list_indices = convert_ROI_table_to_indices(
            original_table,
            level=0,
            coarsening_xy=2,
            full_res_pxl_sizes_zyx=pxl_sizes_zyx,
        )
        region = convert_indices_to_regions(list_indices[0])
        data_array = da.from_zarr(f"{zarr_url}/0")[0]
        img_array = load_region(
            data_zyx=data_array, region=region, compute=True
        )
        assert np.sum(img_array == 0) == 545280

        registered_table = ad.read_zarr(
            f"{zarr_url}/tables/registered_{roi_table}"
        )
        # Create list of indices for 3D ROIs
        list_indices = convert_ROI_table_to_indices(
            registered_table,
            level=0,
            coarsening_xy=2,
            full_res_pxl_sizes_zyx=pxl_sizes_zyx,
        )
        region = convert_indices_to_regions(list_indices[0])
        data_array = da.from_zarr(f"{zarr_url}/0")[0]
        img_array_reg = load_region(
            data_zyx=data_array, region=region, compute=True
        )
        assert np.sum(img_array_reg == 0) == 0

        # Check that the Zarr files contains the relevant label channels:
        with open(f"{zarr_url}/labels/.zattrs", "r") as jsonfile:
            zattrs = json.load(jsonfile)
        assert len(zattrs["labels"]) == 1
        print(zarr_url.split("/")[-1].split("_")[0])
        label_name = f"label_{zarr_url.split('/')[-1].split('_')[0]}_A01_C01"
        assert zattrs["labels"][0] == label_name

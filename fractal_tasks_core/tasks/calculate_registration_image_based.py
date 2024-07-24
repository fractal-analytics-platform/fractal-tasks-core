# Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Tommaso Comparin <tommaso.comparin@exact-lab.it>
# Joel Lüthi <joel.luethi@uzh.ch>
#
# This file is part of Fractal and was originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
"""
Calculates translation for image-based registration
"""
import logging
from enum import Enum

import anndata as ad
import dask.array as da
import numpy as np
import zarr
from pydantic import validate_call
from skimage.registration import phase_cross_correlation

from fractal_tasks_core.channels import get_channel_from_image_zarr
from fractal_tasks_core.channels import OmeroChannel
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.roi import check_valid_ROI_indices
from fractal_tasks_core.roi import (
    convert_indices_to_regions,
)
from fractal_tasks_core.roi import (
    convert_ROI_table_to_indices,
)
from fractal_tasks_core.roi import load_region
from fractal_tasks_core.tables import write_table
from fractal_tasks_core.tasks._registration_utils import (
    calculate_physical_shifts,
)
from fractal_tasks_core.tasks._registration_utils import chi2_shift_out
from fractal_tasks_core.tasks._registration_utils import (
    get_ROI_table_with_translation,
)
from fractal_tasks_core.tasks._registration_utils import is_3D
from fractal_tasks_core.tasks.io_models import InitArgsRegistration

logger = logging.getLogger(__name__)


class RegistrationMethod(Enum):
    """
    RegistrationMethod Enum class

    Attributes:
        PHASE_CROSS_CORRELATION: phase cross correlation based on scikit-image
            (works with 2D & 3D images).
        CHI2_SHIFT: chi2 shift based on image-registration library
            (only works with 2D images).
    """

    PHASE_CROSS_CORRELATION = "phase_cross_correlation"
    CHI2_SHIFT = "chi2_shift"

    def register(self, img_ref, img_acq_x):
        if self == RegistrationMethod.PHASE_CROSS_CORRELATION:
            return phase_cross_correlation(img_ref, img_acq_x)
        elif self == RegistrationMethod.CHI2_SHIFT:
            return chi2_shift_out(img_ref, img_acq_x)


@validate_call
def calculate_registration_image_based(
    *,
    # Fractal arguments
    zarr_url: str,
    init_args: InitArgsRegistration,
    # Core parameters
    wavelength_id: str,
    method: RegistrationMethod = RegistrationMethod.PHASE_CROSS_CORRELATION,
    roi_table: str = "FOV_ROI_table",
    level: int = 2,
) -> None:
    """
    Calculate registration based on images

    This task consists of 3 parts:

    1. Loading the images of a given ROI (=> loop over ROIs)
    2. Calculating the transformation for that ROI
    3. Storing the calculated transformation in the ROI table

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        init_args: Intialization arguments provided by
            `image_based_registration_hcs_init`. They contain the
            reference_zarr_url that is used for registration.
            (standard argument for Fractal tasks, managed by Fractal server).
        wavelength_id: Wavelength that will be used for image-based
            registration; e.g. `A01_C01` for Yokogawa, `C01` for MD.
        method: Method to use for image registration. The available methods
            are `phase_cross_correlation` (scikit-image package, works for 2D
            & 3D) and "chi2_shift" (image_registration package, only works for
            2D images).
        roi_table: Name of the ROI table over which the task loops to
            calculate the registration. Examples: `FOV_ROI_table` => loop over
            the field of views, `well_ROI_table` => process the whole well as
            one image.
        level: Pyramid level of the image to be used for registration.
            Choose `0` to process at full resolution.

    """
    logger.info(
        f"Running for {zarr_url=}.\n"
        f"Calculating translation registration per {roi_table=} for "
        f"{wavelength_id=}."
    )

    init_args.reference_zarr_url = init_args.reference_zarr_url

    # Read some parameters from Zarr metadata
    ngff_image_meta = load_NgffImageMeta(str(init_args.reference_zarr_url))
    coarsening_xy = ngff_image_meta.coarsening_xy

    # Get channel_index via wavelength_id.
    # Intially only allow registration of the same wavelength
    channel_ref: OmeroChannel = get_channel_from_image_zarr(
        image_zarr_path=init_args.reference_zarr_url,
        wavelength_id=wavelength_id,
    )
    channel_index_ref = channel_ref.index

    channel_align: OmeroChannel = get_channel_from_image_zarr(
        image_zarr_path=zarr_url,
        wavelength_id=wavelength_id,
    )
    channel_index_align = channel_align.index

    # Lazily load zarr array
    data_reference_zyx = da.from_zarr(
        f"{init_args.reference_zarr_url}/{level}"
    )[channel_index_ref]
    data_alignment_zyx = da.from_zarr(f"{zarr_url}/{level}")[
        channel_index_align
    ]

    # Check if data is 3D (as not all registration methods work in 3D)
    # TODO: Abstract this check into a higher-level Zarr loading class
    if is_3D(data_reference_zyx):
        if method == RegistrationMethod(RegistrationMethod.CHI2_SHIFT):
            raise ValueError(
                f"The `{RegistrationMethod.CHI2_SHIFT}` registration method "
                "has not been implemented for 3D images and the input image "
                f"had a shape of {data_reference_zyx.shape}."
            )

    # Read ROIs
    ROI_table_ref = ad.read_zarr(
        f"{init_args.reference_zarr_url}/tables/{roi_table}"
    )
    ROI_table_x = ad.read_zarr(f"{zarr_url}/tables/{roi_table}")
    logger.info(
        f"Found {len(ROI_table_x)} ROIs in {roi_table=} to be processed."
    )

    # Check that table type of ROI_table_ref is valid. Note that
    # "ngff:region_table" and None are accepted for backwards compatibility
    valid_table_types = [
        "roi_table",
        "masking_roi_table",
        "ngff:region_table",
        None,
    ]
    ROI_table_ref_group = zarr.open_group(
        f"{init_args.reference_zarr_url}/tables/{roi_table}",
        mode="r",
    )
    ref_table_attrs = ROI_table_ref_group.attrs.asdict()
    ref_table_type = ref_table_attrs.get("type")
    if ref_table_type not in valid_table_types:
        raise ValueError(
            (
                f"Table '{roi_table}' (with type '{ref_table_type}') is "
                "not a valid ROI table."
            )
        )

    # For each acquisition, get the relevant info
    # TODO: Add additional checks on ROIs?
    if (ROI_table_ref.obs.index != ROI_table_x.obs.index).all():
        raise ValueError(
            "Registration is only implemented for ROIs that match between the "
            "acquisitions (e.g. well, FOV ROIs). Here, the ROIs in the "
            f"reference acquisitions were {ROI_table_ref.obs.index}, but the "
            f"ROIs in the alignment acquisition were {ROI_table_x.obs.index}"
        )
    # TODO: Make this less restrictive? i.e. could we also run it if different
    # acquisitions have different FOVs? But then how do we know which FOVs to
    # match?
    # If we relax this, downstream assumptions on matching based on order
    # in the list will break.

    # Read pixel sizes from zarr attributes
    ngff_image_meta_acq_x = load_NgffImageMeta(zarr_url)
    pxl_sizes_zyx = ngff_image_meta.get_pixel_sizes_zyx(level=0)
    pxl_sizes_zyx_acq_x = ngff_image_meta_acq_x.get_pixel_sizes_zyx(level=0)

    if pxl_sizes_zyx != pxl_sizes_zyx_acq_x:
        raise ValueError(
            "Pixel sizes need to be equal between acquisitions for "
            "registration."
        )

    # Create list of indices for 3D ROIs spanning the entire Z direction
    list_indices_ref = convert_ROI_table_to_indices(
        ROI_table_ref,
        level=level,
        coarsening_xy=coarsening_xy,
        full_res_pxl_sizes_zyx=pxl_sizes_zyx,
    )
    check_valid_ROI_indices(list_indices_ref, roi_table)

    list_indices_acq_x = convert_ROI_table_to_indices(
        ROI_table_x,
        level=level,
        coarsening_xy=coarsening_xy,
        full_res_pxl_sizes_zyx=pxl_sizes_zyx,
    )
    check_valid_ROI_indices(list_indices_acq_x, roi_table)

    num_ROIs = len(list_indices_ref)
    compute = True
    new_shifts = {}
    for i_ROI in range(num_ROIs):
        logger.info(
            f"Now processing ROI {i_ROI+1}/{num_ROIs} "
            f"for channel {channel_align}."
        )
        img_ref = load_region(
            data_zyx=data_reference_zyx,
            region=convert_indices_to_regions(list_indices_ref[i_ROI]),
            compute=compute,
        )
        img_acq_x = load_region(
            data_zyx=data_alignment_zyx,
            region=convert_indices_to_regions(list_indices_acq_x[i_ROI]),
            compute=compute,
        )

        ##############
        #  Calculate the transformation
        ##############
        if img_ref.shape != img_acq_x.shape:
            raise NotImplementedError(
                "This registration is not implemented for ROIs with "
                "different shapes between acquisitions."
            )

        shifts = method.register(np.squeeze(img_ref), np.squeeze(img_acq_x))[0]

        ##############
        # Store the calculated transformation ###
        ##############
        # Adapt ROIs for the given ROI table:
        ROI_name = ROI_table_ref.obs.index[i_ROI]
        new_shifts[ROI_name] = calculate_physical_shifts(
            shifts,
            level=level,
            coarsening_xy=coarsening_xy,
            full_res_pxl_sizes_zyx=pxl_sizes_zyx,
        )

    # Write physical shifts to disk (as part of the ROI table)
    logger.info(f"Updating the {roi_table=} with translation columns")
    image_group = zarr.group(zarr_url)
    new_ROI_table = get_ROI_table_with_translation(ROI_table_x, new_shifts)
    write_table(
        image_group,
        roi_table,
        new_ROI_table,
        overwrite=True,
        table_attrs=ref_table_attrs,
    )


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=calculate_registration_image_based,
        logger_name=logger.name,
    )

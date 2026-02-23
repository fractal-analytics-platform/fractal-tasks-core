# Copyright 2022-2026 (C) BioVisionCenter, University of Zurich
"""
Calculates translation for image-based registration
"""

import logging

import numpy as np
from ngio import open_ome_zarr_container
from pydantic import validate_call
from skimage.exposure import rescale_intensity

from fractal_tasks_core._io_models import InitArgsRegistration
from fractal_tasks_core._registration_utils import (
    RegistrationMethod,
    add_translation_to_roi,
)

logger = logging.getLogger("calculate_registration_image_based")


@validate_call
def calculate_registration_image_based(
    *,
    # Fractal arguments
    zarr_url: str,
    init_args: InitArgsRegistration,
    # Core parameters
    wavelength_id: str,
    method: RegistrationMethod = RegistrationMethod.PHASE_CROSS_CORRELATION,
    lower_rescale_quantile: float = 0.0,
    upper_rescale_quantile: float = 0.99,
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
        lower_rescale_quantile: Lower quantile for rescaling the image
            intensities before applying registration. Can be helpful
             to deal with image artifacts. Default is 0.
        upper_rescale_quantile: Upper quantile for rescaling the image
            intensities before applying registration. Can be helpful
            to deal with image artifacts. Default is 0.99.
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

    ref_ome_zarr = open_ome_zarr_container(init_args.reference_zarr_url)
    ref_image = ref_ome_zarr.get_image(path=str(level))
    channel_index_ref = ref_image.get_channel_idx(wavelength_id=wavelength_id)

    to_align_ome_zarr = open_ome_zarr_container(zarr_url)
    to_align_image = to_align_ome_zarr.get_image(path=str(level))
    channel_index_align = to_align_image.get_channel_idx(wavelength_id=wavelength_id)

    if ref_image.is_time_series:
        raise ValueError(
            f"Time series images are currently not supported for image-based "
            f"registration, but the reference image had a time dimension with "
            f"shape {ref_image.shape}."
        )

    if ref_image.is_3d and method == RegistrationMethod.CHI2_SHIFT:
        raise ValueError(
            f"The `{RegistrationMethod.CHI2_SHIFT}` registration method "
            "has not been implemented for 3D images and the input image "
            f"had a shape of {ref_image.shape}."
        )

    ref_roi_table = ref_ome_zarr.get_generic_roi_table(roi_table)
    to_align_roi_table = to_align_ome_zarr.get_generic_roi_table(roi_table)
    logger.info(
        f"Found {len(ref_roi_table.rois())} ROIs in {roi_table=} to be processed."
    )

    to_align_rois = {roi.name: roi for roi in to_align_roi_table.rois()}
    new_shifts = {}
    for roi in ref_roi_table.rois():
        logger.info(
            f"Processing ROI {roi.name!r} for registration between "
            f"{init_args.reference_zarr_url!r} and {zarr_url!r}."
        )
        to_align_roi = to_align_rois.get(roi.name)
        if to_align_roi is None:
            raise ValueError(
                f"Could not find matching ROI for ROI {roi} in the reference "
                "acquisition in the alignment acquisition."
            )
        img_ref = ref_image.get_roi(roi, channel_selection=channel_index_ref)
        img_acq_x = to_align_image.get_roi(
            to_align_roi, channel_selection=channel_index_align
        )

        img_ref = rescale_intensity(
            img_ref,
            in_range=(
                np.quantile(img_ref, lower_rescale_quantile),
                np.quantile(img_ref, upper_rescale_quantile),
            ),
        )
        img_acq_x = rescale_intensity(
            img_acq_x,
            in_range=(
                np.quantile(img_acq_x, lower_rescale_quantile),
                np.quantile(img_acq_x, upper_rescale_quantile),
            ),
        )

        if img_ref.shape != img_acq_x.shape:
            raise NotImplementedError(
                "This registration is not implemented for ROIs with "
                "different shapes between acquisitions."
            )

        shifts = method.register(np.squeeze(img_ref), np.squeeze(img_acq_x))[0]
        new_shifts[roi.name] = shifts

    logger.info(f"Updating the {roi_table=} with translation columns")
    for roi in to_align_roi_table.rois():
        shifts = new_shifts[roi.name]
        updated_roi = add_translation_to_roi(roi, shifts, to_align_image.pixel_size)
        to_align_roi_table.add(updated_roi, overwrite=True)

    to_align_ome_zarr.add_table(
        name=roi_table,
        table=to_align_roi_table,
        overwrite=True,
    )


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=calculate_registration_image_based,
        logger_name=logger.name,
    )

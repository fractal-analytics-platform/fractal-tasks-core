# Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Tommaso Comparin <tommaso.comparin@exact-lab.it>
# Marco Franzon <marco.franzon@exact-lab.it>
#
# This file is part of Fractal and was originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
"""
Apply illumination correction to all fields of view.
"""
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
from ngio import ChannelSelectionModel, open_ome_zarr_container
from ngio.experimental.iterators import ImageProcessingIterator
from pydantic import validate_call
from skimage.io import imread

from fractal_tasks_core.tasks._zarr_utils import _copy_hcs_ome_zarr_metadata

logger = logging.getLogger(__name__)


def correct(
    img_stack: np.ndarray,
    corr_img: np.ndarray,
    dark_img: np.ndarray | None = None,
    background: int = 0,
):
    """
    Corrects a stack of images, using a given illumination profile (e.g. bright
    in the center of the image, dim outside) and a darkfield profile
    (e.g. sensor noise) and the given constant background value.
    Illumination correction is normalized to [0, 1] before application.
    Uses the formula:
        corrected_image = (image - darkfield - background) / illumination

    Args:
        img_stack: 3D numpy array (zyx)
        corr_img: 2D numpy array (yx) with the illumination correction profile.
        dark_img: 2D numpy array (yx) with the darkfield correction profile.
        background: Background value that is subtracted from the image before
            the illumination correction is applied.
    """

    logger.debug(f"Start correct, {img_stack.shape}")

    # Check shapes
    if corr_img.shape != img_stack.shape[-2:]:
        raise ValueError(
            "Error in illumination_correction:\n"
            f"{img_stack.shape=}\n{corr_img.shape=}"
        )
    if dark_img is not None and dark_img.shape != img_stack.shape[-2:]:
        raise ValueError(
            "Error in illumination_correction:\n"
            f"{img_stack.shape=}\n{dark_img.shape=}"
        )

    # Store info about dtype
    dtype = img_stack.dtype
    dtype_max = np.iinfo(dtype).max

    # Background subtraction
    if dark_img is not None:
        logger.debug("Applying darkfield correction")
        img_stack = img_stack.astype(np.int32) - dark_img[None, :, :]
        img_stack[img_stack < 0] = 0
    img_stack[img_stack <= background] = 0
    img_stack[img_stack > background] -= background

    #  Apply the normalized correction matrix (requires a float array)
    # img_stack = img_stack.astype(np.float64)
    img_stack = img_stack / (corr_img / np.max(corr_img))[None, :, :]

    # Handle edge case: corrected image may have values beyond the limit of
    # the encoding, e.g. beyond 65535 for 16bit images. This clips values
    # that surpass this limit and triggers a warning
    if np.any(img_stack > dtype_max):
        logger.warning(
            "Illumination correction created values beyond the max range of "
            f"the current image type. These have been clipped to {dtype_max=}."
        )
        img_stack[img_stack > dtype_max] = dtype_max

    logger.debug("End correct")

    # Cast back to original dtype and return
    return img_stack.astype(dtype)


@validate_call
def illumination_correction(
    *,
    # Fractal parameters
    zarr_url: str,
    # Core parameters
    illumination_profiles_folder: str,
    illumination_profiles: dict[str, str],
    background_profiles_folder: Optional[str] = None,
    background_profiles: Optional[dict[str, str]] = None,
    background: int = 0,
    input_ROI_table: str = "FOV_ROI_table",
    overwrite_input: bool = True,
    # Advanced parameters
    suffix: str = "_illum_corr",
) -> dict[str, Any]:
    """
    Applies illumination correction to the images in the OME-Zarr.

    Assumes that the illumination correction profiles were generated before
    separately and that the same background subtraction was used during
    calculation of the illumination correction (otherwise, it will not work
    well & the correction may only be partial).

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        illumination_profiles_folder: Path of folder of illumination profiles.
        illumination_profiles: Dictionary where keys match the `wavelength_id`
            attributes of existing channels (e.g.  `A01_C01` ) and values are
            the filenames of the corresponding illumination profiles.
        background_profiles_folder: Path of folder of background profiles. If
            not provided, it is assumed that it's the same as
            `illumination_profiles_folder`.
        background_profiles: Dictionary where keys match the `wavelength_id`
            attributes of existing channels (e.g.  `A01_C01` ) and values are
            the filenames of the corresponding background profiles.
            if not provided, no background correction is applied.
        background: Background value that is subtracted from the image before
            the illumination correction is applied. Set it to `0` if you don't
            want any background subtraction.
        input_ROI_table: Name of the ROI table that contains the information
            about the location of the individual field of views (FOVs) to
            which the illumination correction shall be applied. Defaults to
            "FOV_ROI_table", the default name Fractal converters give the ROI
            tables that list all FOVs separately. If you generated your
            OME-Zarr with a different converter and used Import OME-Zarr to
            generate the ROI tables, `image_ROI_table` is the right choice if
            you only have 1 FOV per Zarr image and `grid_ROI_table` if you
            have multiple FOVs per Zarr image and set the right grid options
            during import.
        overwrite_input: If `True`, the results of this task will overwrite
            the input image data. If false, a new image is generated and the
            illumination corrected data is saved there.
        suffix: What suffix to append to the illumination corrected images.
            Only relevant if `overwrite_input=False`.
    """

    # Prepare zarr urls
    zarr_url = zarr_url.rstrip("/")
    if suffix == "":
        raise ValueError("suffix cannot be an empty string.")
    zarr_url_new = f"{zarr_url}{suffix}"

    t_start = time.perf_counter()
    logger.info("Start illumination_correction")
    logger.info(f"  {overwrite_input=}")
    logger.info(f"  {zarr_url=}")
    logger.info(f"  {zarr_url_new=}")

    # Read image metadata
    ome_zarr_container = open_ome_zarr_container(zarr_url)
    image = ome_zarr_container.get_image()

    logger.info(f"{ome_zarr_container=}")
    logger.info(f"{image=}")
    logger.info(f"{image.wavelength_ids=}")

    # Validate that illumination and background profiles cover all wavelengths
    input_image_wavelengths = set(image.wavelength_ids)
    illumination_wavelengths = set(illumination_profiles.keys())
    if not illumination_wavelengths.issubset(input_image_wavelengths):
        raise ValueError(
            "Illumination profiles provided for wavelengths: "
            f"{illumination_wavelengths - input_image_wavelengths} "
            "are not present in the input image."
        )
    if not input_image_wavelengths.issubset(illumination_wavelengths):
        raise ValueError(
            "No illumination profiles provided for wavelengths: "
            f"{input_image_wavelengths - illumination_wavelengths}."
        )
    if background_profiles is not None:
        background_wavelengths = set(background_profiles.keys())
        if not background_wavelengths.issubset(input_image_wavelengths):
            raise ValueError(
                "Background profiles provided for wavelengths: "
                f"{background_wavelengths - input_image_wavelengths} "
                "are not present in the input image."
            )
        if not input_image_wavelengths.issubset(background_wavelengths):
            raise ValueError(
                "No background profiles provided for wavelengths: "
                f"{input_image_wavelengths - background_wavelengths}."
            )
    else:
        background_profiles = {}

    if background_profiles_folder is None:
        background_profiles_folder = illumination_profiles_folder

    # Read FOV ROIs
    FOV_ROI_table = ome_zarr_container.get_generic_roi_table(
        input_ROI_table
    )

    logger.info(f"{FOV_ROI_table=}")

    # Extract image size from FOV-ROI indices. Note: this works at level=0,
    # where FOVs should all be of the exact same size (in pixels)
    image_size = None
    ref_roi_name = None
    for roi in FOV_ROI_table.rois():
        roi_pixels = roi.to_roi_pixels(image.pixel_size)
        if image_size is None:
            image_size = (roi_pixels.y_length, roi_pixels.x_length)
            ref_roi_name = roi.name
        elif (roi_pixels.y_length, roi_pixels.x_length) != image_size:
            raise ValueError(
                "Inconsistent image sizes in the ROI table, found "
                f"{(roi_pixels.y_length, roi_pixels.x_length)} for {roi.name} "
                f"and {image_size} for {ref_roi_name}."
            )
    if image_size is None:
        raise ValueError("No ROIs found in the provided ROI table.")
    
    # Assemble dictionary of correction images and check their shapes
    illumination_corrections = {}
    for wavelength_id, profile_path in illumination_profiles.items():
        correction_matrix = imread(
            (Path(illumination_profiles_folder) / profile_path).as_posix()
        )
        if correction_matrix.shape != image_size:
            raise ValueError(
                f"The illumination {correction_matrix.shape=}"
                f" is different from the input {image_size=}."
                f" for {wavelength_id=}."
            )
        if np.any(correction_matrix == 0):
            raise ValueError(
                f"Illumination correction image for {wavelength_id=} "
                "contains zero values."
            )
        illumination_corrections[wavelength_id] = correction_matrix

    # Assemble dictionary of background images and check their shapes
    background_corrections = {}
    for wavelength_id, profile_path in background_profiles.items():
        correction_matrix = imread(
            (Path(background_profiles_folder) / profile_path).as_posix()
        )
        if correction_matrix.shape != image_size:
            raise ValueError(
                f"The background (darkfield) {correction_matrix.shape=}"
                f" is different from the input {image_size=}"
                f" for {wavelength_id=}."
            )
        background_corrections[wavelength_id] = correction_matrix

    # Create new ome-zarr container for output
    new_ome_zarr = ome_zarr_container.derive_image(
        store=zarr_url_new,
        overwrite=True,
        copy_tables=True,
        copy_labels=True,
    )

    # Start processing loop over channels
    for wavelength_id in image.wavelength_ids:
        logger.info(f"Applying illumination correction for {wavelength_id=}")
        logger.info(f"{image.wavelength_ids=}")
        channel_selection = ChannelSelectionModel(
            mode="wavelength_id", identifier=wavelength_id 
        )
        iterator = ImageProcessingIterator(
            input_image=image,
            output_image=new_ome_zarr.get_image(image.path),
            input_channel_selection=channel_selection,
            output_channel_selection=channel_selection,
        )
        iterator = iterator.product(FOV_ROI_table).by_zyx(strict=False)

        for image_data, writer in iterator.iter_as_numpy():
            writer(
                correct(
                    image_data,
                    illumination_corrections.get(wavelength_id),
                    background_corrections.get(wavelength_id),
                    background=background,
                )
            )

    t_end = time.perf_counter()
    logger.info(f"End illumination_correction, elapsed: {t_end - t_start}")

    if overwrite_input:
        image_list_update = dict(zarr_url=zarr_url)
        # TODO(PR): will not work with s3:// urls, can ngio do this?
        os.rename(zarr_url, f"{zarr_url}_tmp")
        os.rename(zarr_url_new, zarr_url)
        shutil.rmtree(f"{zarr_url}_tmp")
        logger.info("Overwrote input image with illumination corrected image.")
    else:
        # TODO(PR): ngio cannot do this at the moment
        _copy_hcs_ome_zarr_metadata(zarr_url, zarr_url_new)
        image_list_update = dict(zarr_url=zarr_url_new, origin=zarr_url)

    return {"image_list_updates": [image_list_update]}


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=illumination_correction,
        logger_name=logger.name,
    )

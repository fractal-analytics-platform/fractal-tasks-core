# Copyright 2022-2026 (C) BioVisionCenter, University of Zurich
"""Apply illumination correction to all fields of view."""

import logging
import time
from typing import Annotated, Any

import numpy as np
from ngio import ChannelSelectionModel, open_ome_zarr_container, open_ome_zarr_well
from ngio.experimental.iterators import ImageProcessingIterator
from ngio.utils._errors import NgioFileNotFoundError
from pydantic import Field, validate_call
from skimage.io import imread

from fractal_tasks_core._illumination_correction_utils import (
    ConstantCorrectionModel,
    NoCorrectionModel,
    ProfileCorrectionModel,
)
from fractal_tasks_core._utils import format_template_name, split_well_path_image_path

logger = logging.getLogger("illumination_correction")


def correct(
    image: np.ndarray,
    flatfield: np.ndarray,
    darkfield: np.ndarray | None = None,
    background_constant: int = 0,
):
    """Correct a stack of images using illumination and darkfield profiles.

    Uses a given illumination profile (e.g. bright in the center of the
    image, dim outside) and a darkfield profile (e.g. sensor noise) and the
    given constant background value. Illumination correction is normalized
    to [0, 1] before application.
    Uses the formula:
        corrected_image = (image - darkfield - background) / illumination.

    Args:
        image: 3D numpy array (zyx)
        flatfield: 2D numpy array (yx) with the flatfield correction profile.
            Assumed to be normalized to 1 (i.e. max value = 1).
        darkfield: 2D numpy array (yx) with the darkfield correction profile.
        background_constant: Background value that is subtracted from the
            image before the illumination correction is applied.
    """
    logger.debug(f"Start correct, {image.shape}")

    # Check shapes
    if flatfield.shape != image.shape[-2:]:
        raise ValueError(
            f"Error in illumination_correction:\n{image.shape=}\n{flatfield.shape=}"
        )
    if darkfield is not None and darkfield.shape != image.shape[-2:]:
        raise ValueError(
            f"Error in illumination_correction:\n{image.shape=}\n{darkfield.shape=}"
        )

    # Store info about input dtype
    dtype = image.dtype
    dtype_max = np.iinfo(dtype).max

    # Background subtraction
    if darkfield is not None:
        logger.debug("Applying darkfield correction")
        image = (image.astype(np.int32) - darkfield.astype(np.int32)[None, :, :]).clip(
            0, None
        )

    if background_constant != 0:
        logger.debug("Applying constant background correction")
        image = (image.astype(np.int32) - background_constant).clip(0, None)

    #  Apply the normalized correction matrix
    image = image / flatfield[None, :, :]

    # Handle edge case: corrected image may have values beyond the limit of
    # the encoding, e.g. beyond 65535 for 16bit images. This clips values
    # that surpass this limit and triggers a warning
    if np.any(image > dtype_max):
        logger.warning(
            "Illumination correction created values beyond the max range of "
            f"the current image type. These have been clipped to {dtype_max=}."
        )
        image = image.clip(0, dtype_max)

    logger.debug("End correct")

    # Cast back to original dtype and return
    return image.astype(dtype)


BackgroundCorrection = Annotated[
    NoCorrectionModel | ProfileCorrectionModel | ConstantCorrectionModel,
    Field(discriminator="model"),
]


@validate_call
def illumination_correction(
    *,
    # Fractal parameters
    zarr_url: str,
    # Core parameters
    illumination_profiles: ProfileCorrectionModel,
    background_correction: BackgroundCorrection = Field(
        default_factory=NoCorrectionModel
    ),
    input_roi_table: str = "FOV_ROI_table",
    output_image_name: str = Field(
        default="{image_name}_illum_corr",
        pattern=r"^.*\{image_name\}.*$",
    ),
    overwrite_input: bool = False,
) -> dict[str, Any] | None:
    """Applies illumination correction to the images in the OME-Zarr.

    The illumination correction profiles must be pre-computed using the same
    background correction settings as specified here (otherwise the correction
    may only be partial).

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        illumination_profiles: Illumination (flatfield) correction profiles.
        background_correction: (Optional) background (darkfield) correction
            parameters. Can be provided as profiles or as constant values.
        input_roi_table: Name of the ROI table that contains the information
            about the location of the individual field of views (FOVs) to
            which the illumination correction shall be applied. Defaults to
            "FOV_ROI_table", the default name Fractal converters give the ROI
            tables that list all FOVs separately. If you generated your
            OME-Zarr with a different converter and used Import OME-Zarr to
            generate the ROI tables, `image_ROI_table` is the right choice if
            you only have 1 FOV per Zarr image and `grid_ROI_table` if you
            have multiple FOVs per Zarr image and set the right grid options
            during import.
        output_image_name: Name of the output image within the well. Only
            relevant if ``overwrite_input=False``. May contain the placeholder
            ``{image_name}``, which is replaced by the name of the input image
            (i.e. the last path component of ``zarr_url``). For example, the
            default ``"{image_name}_illum_corr"`` turns an input image named
            ``0`` into ``0_illum_corr``.
        overwrite_input: If `True`, the results of this task will overwrite
            the input image data. If `False`, a new image is generated and the
            illumination corrected data is saved there. Defaults to `False`.
    """
    # Prepare zarr urls
    zarr_url = zarr_url.rstrip("/")
    if overwrite_input:
        zarr_url_new = zarr_url
    else:
        image_name = zarr_url.split("/")[-1]
        formatted_name = format_template_name(output_image_name, image_name=image_name)
        parent_url = zarr_url.rsplit("/", 1)[0]
        zarr_url_new = f"{parent_url}/{formatted_name}"

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

    # Validate that provided corrections cover all wavelengths
    input_image_wavelengths = set(image.wavelength_ids)

    illumination_wavelengths = set(illumination_profiles.profiles.keys())
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

    if isinstance(background_correction, ProfileCorrectionModel):
        background_wavelengths = set(background_correction.profiles.keys())

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
    elif isinstance(background_correction, ConstantCorrectionModel):
        background_wavelengths = set(background_correction.constants.keys())
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

    # Read FOV ROIs
    FOV_ROI_table = ome_zarr_container.get_generic_roi_table(input_roi_table)

    logger.info(f"{FOV_ROI_table=}")

    # Extract image size from FOV-ROI indices. Note: this works at level=0,
    # where FOVs should all be of the exact same size (in pixels)
    image_size = None
    ref_roi_name = None
    for roi in FOV_ROI_table.rois():
        roi_pixels = roi.to_pixel(image.pixel_size)
        if image_size is None:
            image_size = (roi_pixels["y"].length, roi_pixels["x"].length)
            ref_roi_name = roi.name
        elif (roi_pixels["y"].length, roi_pixels["x"].length) != image_size:
            raise ValueError(
                "Inconsistent image sizes in the ROI table, found "
                f"{(roi_pixels['y'].length, roi_pixels['x'].length)} for {roi.name} "
                f"and {image_size} for {ref_roi_name}."
            )
    if image_size is None:
        raise ValueError("No ROIs found in the provided ROI table.")

    # Assemble dictionary of correction images and check their shapes
    illumination_matrices = {}
    for wavelength_id, profile_path in illumination_profiles.items():
        correction_matrix = imread(profile_path)
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
        correction_matrix = correction_matrix / np.max(correction_matrix)
        illumination_matrices[wavelength_id] = correction_matrix

    # Assemble dictionary of background images and check their shapes
    background_matrices = {}
    background_constants = {}
    if isinstance(background_correction, ConstantCorrectionModel):
        background_constants = background_correction.constants
    elif isinstance(background_correction, ProfileCorrectionModel):
        for wavelength_id, profile_path in background_correction.items():
            correction_matrix = imread(profile_path)
            if correction_matrix.shape != image_size:
                raise ValueError(
                    f"The background (darkfield) {correction_matrix.shape=}"
                    f" is different from the input {image_size=}"
                    f" for {wavelength_id=}."
                )
            background_matrices[wavelength_id] = correction_matrix

    # Prepare output ome-zarr container and image
    if overwrite_input:
        output_ome_zarr_container = ome_zarr_container
    else:
        # Create new ome-zarr container for output
        output_ome_zarr_container = ome_zarr_container.derive_image(
            store=zarr_url_new,
            overwrite=True,
            copy_tables=True,
            copy_labels=True,
        )
    output_image = output_ome_zarr_container.get_image()

    # Start processing loop over channels
    for wavelength_id in image.wavelength_ids:
        if wavelength_id is None:
            raise ValueError("Image does not contain wavelength_id information.")
        logger.info(f"Applying illumination correction for {wavelength_id=}")
        channel_selection = ChannelSelectionModel(
            mode="wavelength_id", identifier=wavelength_id
        )
        iterator = ImageProcessingIterator(
            input_image=image,
            output_image=output_image,
            input_channel_selection=channel_selection,
            output_channel_selection=channel_selection,
        )
        iterator = iterator.product(FOV_ROI_table).by_zyx(strict=False)
        for image_data, writer in iterator.iter_as_numpy():
            corrected = correct(
                image_data,
                illumination_matrices[wavelength_id],
                background_matrices.get(wavelength_id),
                background_constants.get(wavelength_id, 0),
            )
            # correct function operate on 3D arrays for better performance
            # and in case of 2D, the output will be broadcasted to 3D
            # with a singleton first dimension
            if len(image_data.shape) == 2:
                corrected = corrected.squeeze(axis=0)

            writer(corrected)

    t_end = time.perf_counter()
    logger.info(f"End illumination_correction, elapsed: {t_end - t_start}")

    if not overwrite_input:
        well_url, new_image_path = split_well_path_image_path(zarr_url_new)
        try:
            ome_zarr_well = open_ome_zarr_well(well_url)
            ome_zarr_well.atomic_add_image(
                image_path=new_image_path,
            )
        except NgioFileNotFoundError:
            logger.debug(
                "Input image not in an OME-Zarr Plate, well metadata not updated."
            )

        logger.info(f"Saved illumination corrected image as {zarr_url_new}.")
        image_list_update = {"zarr_url": zarr_url_new, "origin": zarr_url}
        return {"image_list_updates": [image_list_update]}
    else:
        logger.info("Overwrote input image with illumination corrected data.")


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=illumination_correction)

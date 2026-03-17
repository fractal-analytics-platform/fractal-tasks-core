# Copyright 2022-2026 (C) BioVisionCenter, University of Zurich
"""Utils functions for registration"""

from enum import Enum

import numpy as np
from image_registration import chi2_shift
from ngio import PixelSize, Roi
from pydantic import BaseModel
from skimage.registration import phase_cross_correlation


def add_translation_to_roi(roi: Roi, shifts: list[float], pixel_size: PixelSize) -> Roi:
    if len(shifts) == 3:
        shift_dict = {
            "translation_z": float(shifts[0]) * pixel_size.z,
            "translation_y": float(shifts[1]) * pixel_size.y,
            "translation_x": float(shifts[2]) * pixel_size.x,
        }
    elif len(shifts) == 2:
        shift_dict = {
            "translation_z": 0.0,
            "translation_y": float(shifts[0]) * pixel_size.y,
            "translation_x": float(shifts[1]) * pixel_size.x,
        }
    else:
        raise ValueError(f"Wrong input for add_translation_to_roi ({shifts=})")
    return roi.model_copy(update=shift_dict)


def chi2_shift_out(img_ref, img_cycle_x) -> list[np.ndarray]:
    """
    Helper function to get the output of chi2_shift into the same format as
    phase_cross_correlation. Calculates the shift between two images using
    the chi2_shift method.

    Args:
        img_ref (np.ndarray): First image.
        img_cycle_x (np.ndarray): Second image.

    Returns:
        List containing numpy array of shift in y and x direction.
    """
    x, y, a, b = chi2_shift(np.squeeze(img_ref), np.squeeze(img_cycle_x))

    # Running into issues when using direct float output for fractal.
    # When rounding to integer and using integer dtype, it typically works
    # but for some reasons fails when run over a whole 384 well plate (but
    # the well where it fails works fine when run alone). For now, rounding
    # to integer, but still using float64 dtype (like the scikit-image
    # phase cross correlation function) seems to be the safest option.
    shifts = np.array([-np.round(y), -np.round(x)], dtype="float64")
    # return as a list to adhere to the phase_cross_correlation output format
    return [shifts]


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


class InitArgsRegistration(BaseModel):
    """
    Registration init args.

    Passed from `init_image_based_registration` to
    `compute_image_based_registration`.
    """

    reference_zarr_url: str
    """
    zarr_url for the reference image.
    """


class InitArgsRegistrationConsensus(BaseModel):
    """
    Registration consensus init args.

    Provides the list of zarr_urls for all acquisitions for a given well.
    """

    zarr_url_list: list[str]
    """
    List of zarr_urls for all the OME-Zarr images in the well.
    """

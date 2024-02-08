# Copyright 2023 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Joel Lüthi  <joel.luethi@fmi.ch>
#
# This file is part of Fractal and was originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
"""
Helper functions for image normalization in
"""
import logging
from typing import Optional

import numpy as np
from pydantic import BaseModel
from pydantic import Field
from pydantic import root_validator


logger = logging.getLogger(__name__)


class CellposeCustomNormalizer(BaseModel):
    """
    Validator to handle different normalization scenarios for Cellpose models

    If default normalization is to be used, no other parameters can be
    specified. Alternatively, when default normalization is not applied,
    either percentiles or explicit integer bounds can be applied.

    Attributes:
        default_normalize: Whether to use the default Cellpose normalization
            approach (rescaling the image between the 1st and 99th percentile)
        lower_percentile: Specify a custom lower-bound percentile for rescaling
            as a float value between 0 and 100. Set to 1 to run the same as
            default). You can only specify percentils or bounds, not both.
        upper_percentile: Specify a custom upper-bound percentile for rescaling
            as a float value between 0 and 100. Set to 99 to run the same as
            default, set to e.g. 99.99 if the default rescaling was too harsh.
            You can only specify percentils or bounds, not both.
        lower_bound: Explicit lower bound value to rescale the image at.
            Needs to be an integer, e.g. 100.
            You can only specify percentils or bounds, not both.
        upper_bound: Explicit upper bound value to rescale the image at.
            Needs to be an integer, e.g. 2000
            You can only specify percentils or bounds, not both.

    """

    default_normalize: bool = True
    lower_percentile: Optional[float] = Field(None, ge=0, le=100)
    upper_percentile: Optional[float] = Field(None, ge=0, le=100)
    lower_bound: Optional[int] = None
    upper_bound: Optional[int] = None

    # In the future, add an option to allow using precomputed percentiles
    # that are stored in OME-Zarr histograms and use this pydantic model that
    # those histograms actually exist

    @root_validator
    def validate_conditions(cls, values):
        default_normalize = values.get("default_normalize")
        lower_percentile = values.get("lower_percentile")
        upper_percentile = values.get("upper_percentile")
        lower_bound = values.get("lower_bound")
        upper_bound = values.get("upper_bound")

        # If default_normalize is True, check that all other fields are None
        if default_normalize:
            if any(
                v is not None
                for v in [
                    lower_percentile,
                    upper_percentile,
                    lower_bound,
                    upper_bound,
                ]
            ):
                raise ValueError(
                    "When default_normalize is True, no percentile or "
                    "bounds can be specified"
                )

        # Check for lower_percentile and upper_percentile condition
        if lower_percentile is not None or upper_percentile is not None:
            if lower_percentile is None or upper_percentile is None:
                raise ValueError(
                    "Both lower_percentile and upper_percentile must be set "
                    "together"
                )
            if lower_bound is not None or upper_bound is not None:
                raise ValueError(
                    "If a percentile is specified, no hard set lower_bound "
                    "or upper_bound can be specified"
                )

        # If lower_bound or upper_bound is set, the other must also be set,
        # and lower_percentile & upper_percentile must be None
        if lower_bound is not None or upper_bound is not None:
            if lower_bound is None or upper_bound is None:
                raise ValueError(
                    "Both lower_bound and upper_bound must be set together"
                )
            if lower_percentile is not None or upper_percentile is not None:
                raise ValueError(
                    "If explicit lower and upper bounds are set, percentiles "
                    "cannot be specified"
                )

        return values


def normalized_img(
    img,
    axis=-1,
    invert=False,
    lower_p: float = 1.0,
    upper_p: float = 99.0,
    lower_bound=None,
    upper_bound=None,
):
    """normalize each channel of the image so that so that 0.0=lower percentile
    or lower bound and 1.0=upper percentile or upper bound of image intensities.

    The normalization can result in values < 0 or > 1 (no clipping).

    Based on https://github.com/MouseLand/cellpose/blob/4f5661983c3787efa443bbbd3f60256f4fd8bf53/cellpose/transforms.py#L375 # noqa E501

    optional inversion

    Parameters
    ------------

    img: ND-array (at least 3 dimensions)

    axis: channel axis to loop over for normalization

    invert: invert image (useful if cells are dark instead of bright)

    lower_p: Lower percentile for rescaling

    upper_p: Upper percentile for rescaling

    lower_bound: Lower fixed-value used for rescaling

    upper_bound: Upper fixed-value used for rescaling

    Returns
    ---------------

    img: ND-array, float32
        normalized image of same size

    """
    if img.ndim < 3:
        error_message = "Image needs to have at least 3 dimensions"
        logger.critical(error_message)
        raise ValueError(error_message)

    img = img.astype(np.float32)
    img = np.moveaxis(img, axis, 0)
    for k in range(img.shape[0]):
        if lower_p is not None:
            # ptp can still give nan's with weird images
            i99 = np.percentile(img[k], upper_p)
            i1 = np.percentile(img[k], lower_p)
            if i99 - i1 > +1e-3:  # np.ptp(img[k]) > 1e-3:
                img[k] = normalize_percentile(
                    img[k], lower=lower_p, upper=upper_p
                )
                if invert:
                    img[k] = -1 * img[k] + 1
            else:
                img[k] = 0
        elif lower_bound is not None:
            if upper_bound - lower_bound > +1e-3:
                img[k] = normalize_bounds(
                    img[k], lower=lower_bound, upper=upper_bound
                )
                if invert:
                    img[k] = -1 * img[k] + 1
            else:
                img[k] = 0
        else:
            raise ValueError("No normalization method specified")
    img = np.moveaxis(img, 0, axis)
    return img


def normalize_percentile(Y, lower: int = 1, upper: int = 99):
    """normalize image so 0.0 is lower percentile and 1.0 is upper percentile
    Percentiles are passed as integers
    """
    X = Y.copy()
    x01 = np.percentile(X, lower)
    x99 = np.percentile(X, upper)
    X = (X - x01) / (x99 - x01)
    return X


def normalize_bounds(Y, lower: int = 0, upper: int = 65535):
    """normalize image so 0.0 is lower percentile and 1.0 is upper percentile
    Percentiles are passed as integers
    """
    X = Y.copy()
    X = (X - lower) / (upper - lower)
    return X

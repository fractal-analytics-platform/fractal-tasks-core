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
from typing import Literal
from typing import Optional

import numpy as np
from pydantic import BaseModel
from pydantic import Field
from pydantic import root_validator


logger = logging.getLogger(__name__)


class CellposeCustomNormalizer(BaseModel):
    """
    Validator to handle different normalization scenarios for Cellpose models

    If `type="default"`, then Cellpose default normalization is
    used and no other parameters can be specified.
    If `type="no_normalization"`, then no normalization is used and no
    other parameters can be specified.
    If `type="custom"`, then either percentiles or explicit integer
    bounds can be applied.

    Attributes:
        type:
            One of `default` (Cellpose default normalization), `custom`
            (using the other custom parameters) or `no_normalization`.
        lower_percentile: Specify a custom lower-bound percentile for rescaling
            as a float value between 0 and 100. Set to 1 to run the same as
            default). You can only specify percentiles or bounds, not both.
        upper_percentile: Specify a custom upper-bound percentile for rescaling
            as a float value between 0 and 100. Set to 99 to run the same as
            default, set to e.g. 99.99 if the default rescaling was too harsh.
            You can only specify percentiles or bounds, not both.
        lower_bound: Explicit lower bound value to rescale the image at.
            Needs to be an integer, e.g. 100.
            You can only specify percentiles or bounds, not both.
        upper_bound: Explicit upper bound value to rescale the image at.
            Needs to be an integer, e.g. 2000.
            You can only specify percentiles or bounds, not both.
    """

    type: Literal["default", "custom", "no_normalization"] = "default"
    lower_percentile: Optional[float] = Field(None, ge=0, le=100)
    upper_percentile: Optional[float] = Field(None, ge=0, le=100)
    lower_bound: Optional[int] = None
    upper_bound: Optional[int] = None

    # In the future, add an option to allow using precomputed percentiles
    # that are stored in OME-Zarr histograms and use this pydantic model that
    # those histograms actually exist

    @root_validator
    def validate_conditions(cls, values):
        # Extract values
        type = values.get("type")
        lower_percentile = values.get("lower_percentile")
        upper_percentile = values.get("upper_percentile")
        lower_bound = values.get("lower_bound")
        upper_bound = values.get("upper_bound")

        # Verify that custom parameters are only provided when type="custom"
        if type != "custom":
            if lower_percentile is not None:
                raise ValueError(
                    f"Type='{type}' but {lower_percentile=}. "
                    "Hint: set type='custom'."
                )
            if upper_percentile is not None:
                raise ValueError(
                    f"Type='{type}' but {upper_percentile=}. "
                    "Hint: set type='custom'."
                )
            if lower_bound is not None:
                raise ValueError(
                    f"Type='{type}' but {lower_bound=}. "
                    "Hint: set type='custom'."
                )
            if upper_bound is not None:
                raise ValueError(
                    f"Type='{type}' but {upper_bound=}. "
                    "Hint: set type='custom'."
                )

        # The only valid options are:
        # 1. Both percentiles are set and both bounds are unset
        # 2. Both bounds are set and both percentiles are unset
        are_percentiles_set = (
            lower_percentile is not None,
            upper_percentile is not None,
        )
        are_bounds_set = (
            lower_bound is not None,
            upper_bound is not None,
        )
        if len(set(are_percentiles_set)) != 1:
            raise ValueError(
                "Both lower_percentile and upper_percentile must be set "
                "together."
            )
        if len(set(are_bounds_set)) != 1:
            raise ValueError(
                "Both lower_bound and upper_bound must be set together"
            )
        if lower_percentile is not None and lower_bound is not None:
            raise ValueError(
                "You cannot set both explicit bounds and percentile bounds "
                "at the same time. Hint: use only one of the two options."
            )

        return values

    @property
    def cellpose_normalize(self) -> bool:
        """
        Determine whether cellpose should apply its internal normalization.

        If type is set to `custom` or `no_normalization`, don't apply cellpose
        internal normalization
        """
        return self.type == "default"


def normalized_img(
    img: np.ndarray,
    axis: int = -1,
    invert: bool = False,
    lower_p: float = 1.0,
    upper_p: float = 99.0,
    lower_bound: Optional[int] = None,
    upper_bound: Optional[int] = None,
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


def normalize_percentile(Y: np.ndarray, lower: float = 1, upper: float = 99):
    """normalize image so 0.0 is lower percentile and 1.0 is upper percentile
    Percentiles are passed as floats (must be between 0 and 100)

    Args:
        Y: The image to be normalized
        lower: Lower percentile
        upper: Upper percentile

    """
    X = Y.copy()
    x01 = np.percentile(X, lower)
    x99 = np.percentile(X, upper)
    X = (X - x01) / (x99 - x01)
    return X


def normalize_bounds(Y: np.ndarray, lower: int = 0, upper: int = 65535):
    """normalize image so 0.0 is lower value and 1.0 is upper value

    Args:
        Y: The image to be normalized
        lower: Lower normalization value
        upper: Upper normalization value

    """
    X = Y.copy()
    X = (X - lower) / (upper - lower)
    return X

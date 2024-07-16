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
from pydantic import model_validator
from typing_extensions import Self

from fractal_tasks_core.channels import ChannelInputModel
from fractal_tasks_core.channels import ChannelNotFoundError
from fractal_tasks_core.channels import get_channel_from_image_zarr
from fractal_tasks_core.channels import OmeroChannel


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

    @model_validator(mode="after")
    def validate_conditions(self: Self) -> Self:
        # Extract values
        type = self.type
        lower_percentile = self.lower_percentile
        upper_percentile = self.upper_percentile
        lower_bound = self.lower_bound
        upper_bound = self.upper_bound

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

        return self

    @property
    def cellpose_normalize(self) -> bool:
        """
        Determine whether cellpose should apply its internal normalization.

        If type is set to `custom` or `no_normalization`, don't apply cellpose
        internal normalization
        """
        return self.type == "default"


class CellposeModelParams(BaseModel):
    """
    Advanced Cellpose Model Parameters

    Attributes:
        cellprob_threshold: Parameter of `CellposeModel.eval` method. Valid
            values between -6 to 6. From Cellpose documentation: "Decrease this
            threshold if cellpose is not returning as many ROIs as you'd
            expect. Similarly, increase this threshold if cellpose is returning
            too ROIs particularly from dim areas."
        flow_threshold: Parameter of `CellposeModel.eval` method. Valid
            values between 0.0 and 1.0. From Cellpose documentation: "Increase
            this threshold if cellpose is not returning as many ROIs as you'd
            expect. Similarly, decrease this threshold if cellpose is returning
            too many ill-shaped ROIs."
        anisotropy: Ratio of the pixel sizes along Z and XY axis (ignored if
            the image is not three-dimensional). If unset, it is inferred from
            the OME-NGFF metadata.
        min_size: Parameter of `CellposeModel` class. Minimum size of the
            segmented objects (in pixels). Use `-1` to turn off the size
            filter.
        augment: Parameter of `CellposeModel` class. Whether to use cellpose
            augmentation to tile images with overlap.
        net_avg: Parameter of `CellposeModel` class. Whether to use cellpose
            net averaging to run the 4 built-in networks (useful for `nuclei`,
            `cyto` and `cyto2`, not sure it works for the others).
        use_gpu: If `False`, always use the CPU; if `True`, use the GPU if
            possible (as defined in `cellpose.core.use_gpu()`) and fall-back
            to the CPU otherwise.
        batch_size: number of 224x224 patches to run simultaneously on the GPU
            (can make smaller or bigger depending on GPU memory usage)
        invert: invert image pixel intensity before running network (if True,
            image is also normalized)
        tile: tiles image to ensure GPU/CPU memory usage limited (recommended)
        tile_overlap: fraction of overlap of tiles when computing flows
        resample: run dynamics at original image size (will be slower but
            create more accurate boundaries)
        interp: interpolate during 2D dynamics (not available in 3D)
            (in previous versions it was False, now it defaults to True)
        stitch_threshold: if stitch_threshold>0.0 and not do_3D and equal
            image sizes, masks are stitched in 3D to return volume segmentation
    """

    cellprob_threshold: float = 0.0
    flow_threshold: float = 0.4
    anisotropy: Optional[float] = None
    min_size: int = 15
    augment: bool = False
    net_avg: bool = False
    use_gpu: bool = True
    batch_size: int = 8
    invert: bool = False
    tile: bool = True
    tile_overlap: float = 0.1
    resample: bool = True
    interp: bool = True
    stitch_threshold: float = 0.0


class CellposeChannel1InputModel(ChannelInputModel):
    """
    Channel input for cellpose with normalization options.

    Attributes:
        wavelength_id: Unique ID for the channel wavelength, e.g. `A01_C01`.
            Can only be specified if label is not set.
        label: Name of the channel. Can only be specified if wavelength_id is
            not set.
        normalize: Validator to handle different normalization scenarios for
            Cellpose models
    """

    normalize: CellposeCustomNormalizer = Field(
        default_factory=CellposeCustomNormalizer
    )

    def get_omero_channel(self, zarr_url) -> OmeroChannel:
        try:
            return get_channel_from_image_zarr(
                image_zarr_path=zarr_url,
                wavelength_id=self.wavelength_id,
                label=self.label,
            )
        except ChannelNotFoundError as e:
            logger.warning(
                f"Channel with wavelength_id: {self.wavelength_id} "
                f"and label: {self.label} not found, exit from the task.\n"
                f"Original error: {str(e)}"
            )
            return None


class CellposeChannel2InputModel(BaseModel):
    """
    Channel input for secondary cellpose channel with normalization options.

    The secondary channel is Optional, thus both wavelength_id and label are
    optional to be set. The `is_set` function shows whether either value was
    set.

    Attributes:
        wavelength_id: Unique ID for the channel wavelength, e.g. `A01_C01`.
            Can only be specified if label is not set.
        label: Name of the channel. Can only be specified if wavelength_id is
            not set.
        normalize: Validator to handle different normalization scenarios for
            Cellpose models
    """

    wavelength_id: Optional[str] = None
    label: Optional[str] = None
    normalize: CellposeCustomNormalizer = Field(
        default_factory=CellposeCustomNormalizer
    )

    @model_validator(mode="after")
    def mutually_exclusive_channel_attributes(self: Self) -> Self:
        """
        Check that only 1 of `label` or `wavelength_id` is set.
        """
        wavelength_id = self.wavelength_id
        label = self.label
        if (wavelength_id is not None) and (label is not None):
            raise ValueError(
                "`wavelength_id` and `label` cannot be both set "
                f"(given {wavelength_id=} and {label=})."
            )
        return self

    def is_set(self):
        if self.wavelength_id or self.label:
            return True
        return False

    def get_omero_channel(self, zarr_url) -> OmeroChannel:
        try:
            return get_channel_from_image_zarr(
                image_zarr_path=zarr_url,
                wavelength_id=self.wavelength_id,
                label=self.label,
            )
        except ChannelNotFoundError as e:
            logger.warning(
                f"Second channel with wavelength_id: {self.wavelength_id} "
                f"and label: {self.label} not found, exit from the task.\n"
                f"Original error: {str(e)}"
            )
            return None


def _normalize_cellpose_channels(
    x: np.ndarray,
    channels: list[int],
    normalize: CellposeCustomNormalizer,
    normalize2: CellposeCustomNormalizer,
) -> np.ndarray:
    """
    Normalize a cellpose input array by channel.

    Args:
        x: 4D numpy array.
        channels: Which channels to use. If only one channel is provided, `[0,
            0]` should be used. If two channels are provided (the first
            dimension of `x` has length of 2), `[1, 2]` should be used
            (`x[0, :, :,:]` contains the membrane channel and
            `x[1, :, :, :]` contains the nuclear channel).
        normalize: By default, data is normalized so 0.0=1st percentile and
            1.0=99th percentile of image intensities in each channel.
            This automatic normalization can lead to issues when the image to
            be segmented is very sparse. You can turn off the default
            rescaling. With the "custom" option, you can either provide your
            own rescaling percentiles or fixed rescaling upper and lower
            bound integers.
        normalize2: Normalization options for channel 2. If one channel is
            normalized with default settings, both channels need to be
            normalized with default settings.

    """
    # Optionally perform custom normalization
    # normalize channels separately, if normalize2 is provided:
    if channels == [1, 2]:
        # If run in single channel mode, fails (specified as channel = [0, 0])
        if (normalize.type == "default") != (normalize2.type == "default"):
            raise ValueError(
                "ERROR in normalization:"
                f" {normalize.type=} and {normalize2.type=}."
                " Either both need to be 'default', or none of them."
            )
        if normalize.type == "custom":
            x[channels[0] - 1 : channels[0]] = normalized_img(
                x[channels[0] - 1 : channels[0]],
                lower_p=normalize.lower_percentile,
                upper_p=normalize.upper_percentile,
                lower_bound=normalize.lower_bound,
                upper_bound=normalize.upper_bound,
            )
        if normalize2.type == "custom":
            x[channels[1] - 1 : channels[1]] = normalized_img(
                x[channels[1] - 1 : channels[1]],
                lower_p=normalize2.lower_percentile,
                upper_p=normalize2.upper_percentile,
                lower_bound=normalize2.lower_bound,
                upper_bound=normalize2.upper_bound,
            )

    # otherwise, use first normalize to normalize all channels:
    else:
        if normalize.type == "custom":
            x = normalized_img(
                x,
                lower_p=normalize.lower_percentile,
                upper_p=normalize.upper_percentile,
                lower_bound=normalize.lower_bound,
                upper_bound=normalize.upper_bound,
            )

    return x


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

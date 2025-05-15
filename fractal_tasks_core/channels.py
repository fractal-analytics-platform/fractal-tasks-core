# Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Tommaso Comparin <tommaso.comparin@exact-lab.it>
#
# This file is part of Fractal and was originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
"""
Helper functions to address channels via OME-NGFF/OMERO metadata.
"""
import logging
from copy import deepcopy
from typing import Any
from typing import Optional
from typing import Union

import zarr
from pydantic import BaseModel
from pydantic import field_validator
from pydantic import model_validator
from typing_extensions import Self

from fractal_tasks_core import __OME_NGFF_VERSION__


if __OME_NGFF_VERSION__ != "0.4":
    NotImplementedError(
        f"OME NGFF {__OME_NGFF_VERSION__} is not supported " "in `channels.py`"
    )


class Window(BaseModel):
    """
    Custom class for Omero-channel window, based on OME-NGFF v0.4.

    Attributes:
        min: Do not change. It will be set to `0` by default.
        max:
            Do not change. It will be set according to bit-depth of the images
            by default (e.g. 65535 for 16 bit images).
        start: Lower-bound rescaling value for visualization.
        end: Upper-bound rescaling value for visualization.
    """

    min: Optional[int] = None
    max: Optional[int] = None
    start: int
    end: int


class OmeroChannel(BaseModel):
    """
    Custom class for Omero channels, based on OME-NGFF v0.4.

    Attributes:
        wavelength_id: Unique ID for the channel wavelength, e.g. `A01_C01`.
        index: Do not change. For internal use only.
        label: Name of the channel.
        window: Optional `Window` object to set default display settings. If
            unset, it will be set to the full bit range of the image
            (e.g. 0-65535 for 16 bit images).
        color: Optional hex colormap to display the channel in napari (it
            must be of length 6, e.g. `00FFFF`).
        active: Should this channel be shown in the viewer?
        coefficient: Do not change. Omero-channel attribute.
        inverted: Do not change. Omero-channel attribute.
    """

    # Custom

    wavelength_id: str
    index: Optional[int] = None

    # From OME-NGFF v0.4 transitional metadata

    label: Optional[str] = None
    window: Optional[Window] = None
    color: Optional[str] = None
    active: bool = True
    coefficient: int = 1
    inverted: bool = False

    @field_validator("color", mode="after")
    @classmethod
    def valid_hex_color(cls, v: Optional[str]) -> Optional[str]:
        """
        Check that `color` is made of exactly six elements which are letters
        (a-f or A-F) or digits (0-9).
        """
        if v is None:
            return v
        if len(v) != 6:
            raise ValueError(f'color must have length 6 (given: "{v}")')
        allowed_characters = "abcdefABCDEF0123456789"
        for character in v:
            if character not in allowed_characters:
                raise ValueError(
                    "color must only include characters from "
                    f'"{allowed_characters}" (given: "{v}")'
                )
        return v


class ChannelInputModel(BaseModel):
    """
    A channel which is specified by either `wavelength_id` or `label`.

    This model is similar to `OmeroChannel`, but it is used for
    task-function arguments (and for generating appropriate JSON schemas).

    Attributes:
        wavelength_id: Unique ID for the channel wavelength, e.g. `A01_C01`.
            Can only be specified if label is not set.
        label: Name of the channel. Can only be specified if wavelength_id is
            not set.
    """

    wavelength_id: Optional[str] = None
    label: Optional[str] = None

    @model_validator(mode="after")
    def mutually_exclusive_channel_attributes(self: Self) -> Self:
        """
        Check that either `label` or `wavelength_id` is set.
        """
        wavelength_id = self.wavelength_id
        label = self.label

        if wavelength_id and label:
            raise ValueError(
                "`wavelength_id` and `label` cannot be both set "
                f"(given {wavelength_id=} and {label=})."
            )
        if wavelength_id is None and label is None:
            raise ValueError(
                "`wavelength_id` and `label` cannot be both `None`"
            )
        return self


class ChannelNotFoundError(ValueError):
    """
    Custom error for when `get_channel_from_list` fails,
    that can be captured and handled upstream if needed.
    """

    pass


def check_unique_wavelength_ids(channels: list[OmeroChannel]):
    """
    Check that the `wavelength_id` attributes of a channel list are unique.

    Args:
        channels: TBD
    """
    wavelength_ids = [c.wavelength_id for c in channels]
    if len(set(wavelength_ids)) < len(wavelength_ids):
        raise ValueError(
            f"Non-unique wavelength_id's in {wavelength_ids}\n" f"{channels=}"
        )


def check_well_channel_labels(*, well_zarr_path: str) -> None:
    """
    Check that the channel labels for a well are unique.

    First identify the channel-labels list for each image in the well, then
    compare lists and verify their intersection is empty.

    Args:
        well_zarr_path: path to an OME-NGFF well zarr group.
    """

    # Iterate over all images (multiplexing acquisitions, multi-FOVs, ...)
    group = zarr.open_group(well_zarr_path, mode="r+")
    image_paths = [image["path"] for image in group.attrs["well"]["images"]]
    list_of_channel_lists = []
    for image_path in image_paths:
        channels = get_omero_channel_list(
            image_zarr_path=f"{well_zarr_path}/{image_path}"
        )
        list_of_channel_lists.append(channels[:])

    # For each pair of channel-labels lists, verify they do not overlap
    for ind_1, channels_1 in enumerate(list_of_channel_lists):
        labels_1 = set([c.label for c in channels_1])
        for ind_2 in range(ind_1):
            channels_2 = list_of_channel_lists[ind_2]
            labels_2 = set([c.label for c in channels_2])
            intersection = labels_1 & labels_2
            if intersection:
                hint = (
                    "Are you parsing fields of view into separate OME-Zarr "
                    "images? This could lead to non-unique channel labels, "
                    "and then could be the reason of the error"
                )
                raise ValueError(
                    "Non-unique channel labels\n"
                    f"{labels_1=}\n{labels_2=}\n{hint}"
                )


def get_channel_from_image_zarr(
    *,
    image_zarr_path: str,
    label: Optional[str] = None,
    wavelength_id: Optional[str] = None,
) -> OmeroChannel:
    """
    Extract a channel from OME-NGFF zarr attributes.

    This is a helper function that combines `get_omero_channel_list` with
    `get_channel_from_list`.

    Args:
        image_zarr_path: Path to an OME-NGFF image zarr group.
        label: `label` attribute of the channel to be extracted.
        wavelength_id: `wavelength_id` attribute of the channel to be
            extracted.

    Returns:
        A single channel dictionary.
    """
    omero_channels = get_omero_channel_list(image_zarr_path=image_zarr_path)
    channel = get_channel_from_list(
        channels=omero_channels, label=label, wavelength_id=wavelength_id
    )
    return channel


def get_omero_channel_list(*, image_zarr_path: str) -> list[OmeroChannel]:
    """
    Extract the list of channels from OME-NGFF zarr attributes.

    Args:
        image_zarr_path: Path to an OME-NGFF image zarr group.

    Returns:
        A list of channel dictionaries.
    """
    group = zarr.open_group(image_zarr_path, mode="r+")
    channels_dicts = group.attrs["omero"]["channels"]
    channels = [OmeroChannel(**c) for c in channels_dicts]
    return channels


def get_channel_from_list(
    *,
    channels: list[OmeroChannel],
    label: Optional[str] = None,
    wavelength_id: Optional[str] = None,
) -> OmeroChannel:
    """
    Find matching channel in a list.

    Find the channel that has the required values of `label` and/or
    `wavelength_id`, and identify its positional index (which also
    corresponds to its index in the zarr array).

    Args:
        channels: A list of channel dictionary, where each channel includes (at
            least) the `label` and `wavelength_id` keys.
        label: The label to look for in the list of channels.
        wavelength_id: The wavelength_id to look for in the list of channels.

    Returns:
        A single channel dictionary.
    """

    # Identify matching channels
    if label:
        if wavelength_id:
            # Both label and wavelength_id are specified
            matching_channels = [
                c
                for c in channels
                if (c.label == label and c.wavelength_id == wavelength_id)
            ]
        else:
            # Only label is specified
            matching_channels = [c for c in channels if c.label == label]
    else:
        if wavelength_id:
            # Only wavelength_id is specified
            matching_channels = [
                c for c in channels if c.wavelength_id == wavelength_id
            ]
        else:
            # Neither label or wavelength_id are specified
            raise ValueError(
                "get_channel requires at least one in {label,wavelength_id} "
                "arguments"
            )

    # Verify that there is one and only one matching channel
    if len(matching_channels) == 0:
        required_match = [f"{label=}", f"{wavelength_id=}"]
        required_match_string = " and ".join(
            [x for x in required_match if "None" not in x]
        )
        raise ChannelNotFoundError(
            f"ChannelNotFoundError: No channel found in {channels}"
            f" for {required_match_string}"
        )
    if len(matching_channels) > 1:
        raise ValueError(f"Inconsistent set of channels: {channels}")

    channel = matching_channels[0]
    channel.index = channels.index(channel)
    return channel


def define_omero_channels(
    *,
    channels: list[OmeroChannel],
    bit_depth: int,
    label_prefix: Optional[str] = None,
) -> list[dict[str, Union[str, int, bool, dict[str, int]]]]:
    """
    Update a channel list to use it in the OMERO/channels metadata.

    Given a list of channel dictionaries, update each one of them by:
        1. Adding a label (if missing);
        2. Adding a set of OMERO-specific attributes;
        3. Discarding all other attributes.

    The `new_channels` output can be used in the `attrs["omero"]["channels"]`
    attribute of an image group.

    Args:
        channels: A list of channel dictionaries (each one must include the
            `wavelength_id` key).
        bit_depth: bit depth.
        label_prefix: Prefix to be added before the default label. Used e.g.
            to add a prefix for the acquisition round.

    Returns:
        `new_channels`, a new list of consistent channel dictionaries that
            can be written to OMERO metadata.
    """

    new_channels = [c.model_copy(deep=True) for c in channels]
    default_colors = ["00FFFF", "FF00FF", "FFFF00"]

    for channel in new_channels:
        wavelength_id = channel.wavelength_id

        # If channel.label is None, set it to a default value
        if channel.label is None:
            default_label = wavelength_id
            if label_prefix is not None:
                default_label = f"{label_prefix}_{default_label}"
            logging.warning(
                f"Missing label for {channel=}, using {default_label=}"
            )
            channel.label = default_label

        # If channel.color is None, set it to a default value (use the default
        # ones for the first three channels, or gray otherwise)
        if channel.color is None:
            try:
                channel.color = default_colors.pop()
            except IndexError:
                channel.color = "808080"

        # Set channel.window attribute
        if channel.window:
            channel.window.min = 0
            channel.window.max = 2**bit_depth - 1
        else:
            # If no channel.window is set, create a new one with full bitrange
            # min & max
            channel.window = Window(
                min=0,
                max=2**bit_depth - 1,
                start=0,
                end=2**bit_depth - 1,
            )

    # Check that channel labels are unique for this image
    labels = [c.label for c in new_channels]
    if len(set(labels)) < len(labels):
        raise ValueError(f"Non-unique labels in {new_channels=}")

    new_channels_dictionaries = [
        c.model_dump(exclude={"index"}, exclude_unset=True)
        for c in new_channels
    ]

    return new_channels_dictionaries


def _get_new_unique_value(
    value: str,
    existing_values: list[str],
) -> str:
    """
    Produce a string value that is not present in a given list

    Append `_1`, `_2`, ... to a given string, if needed, until finding a value
    which is not already present in `existing_values`.

    Args:
        value: The first guess for the new value
        existing_values: The list of existing values

    Returns:
        A string value which is not present in `existing_values`
    """
    counter = 1
    new_value = value
    while new_value in existing_values:
        new_value = f"{value}-{counter}"
        counter += 1
    return new_value


def update_omero_channels(
    old_channels: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """
    Make an existing list of Omero channels Fractal-compatible

    The output channels all have keys `label`, `wavelength_id` and `color`;
    the `wavelength_id` values are unique across the channel list.

    See https://ngff.openmicroscopy.org/0.4/index.html#omero-md for the
    definition of NGFF Omero metadata.

    Args:
        old_channels: Existing list of Omero-channel dictionaries

    Returns:
        New list of Fractal-compatible Omero-channel dictionaries
    """
    new_channels = deepcopy(old_channels)
    existing_wavelength_ids: list[str] = []
    handled_channels = []

    default_colors = ["00FFFF", "FF00FF", "FFFF00"]

    def _get_next_color() -> str:
        try:
            return default_colors.pop(0)
        except IndexError:
            return "808080"

    # Channels that contain the key "wavelength_id"
    for ind, old_channel in enumerate(old_channels):
        if "wavelength_id" in old_channel.keys():
            handled_channels.append(ind)
            existing_wavelength_ids.append(old_channel["wavelength_id"])
            new_channel = old_channel.copy()
            try:
                label = old_channel["label"]
            except KeyError:
                label = str(ind + 1)
            new_channel["label"] = label
            if "color" not in old_channel:
                new_channel["color"] = _get_next_color()
            new_channels[ind] = new_channel

    # Channels that contain the key "label" but do not contain the key
    # "wavelength_id"
    for ind, old_channel in enumerate(old_channels):
        if ind in handled_channels:
            continue
        if "label" not in old_channel.keys():
            continue
        handled_channels.append(ind)
        label = old_channel["label"]
        wavelength_id = _get_new_unique_value(
            label,
            existing_wavelength_ids,
        )
        existing_wavelength_ids.append(wavelength_id)
        new_channel = old_channel.copy()
        new_channel["wavelength_id"] = wavelength_id
        if "color" not in old_channel:
            new_channel["color"] = _get_next_color()
        new_channels[ind] = new_channel

    # Channels that do not contain the key "label" nor the key "wavelength_id"
    # NOTE: these channels must be treated last, as they have lower priority
    # w.r.t. existing "wavelength_id" or "label" values
    for ind, old_channel in enumerate(old_channels):
        if ind in handled_channels:
            continue
        label = str(ind + 1)
        wavelength_id = _get_new_unique_value(
            label,
            existing_wavelength_ids,
        )
        existing_wavelength_ids.append(wavelength_id)
        new_channel = old_channel.copy()
        new_channel["label"] = label
        new_channel["wavelength_id"] = wavelength_id
        if "color" not in old_channel:
            new_channel["color"] = _get_next_color()
        new_channels[ind] = new_channel

    # Log old/new values of label, wavelength_id and color
    for ind, old_channel in enumerate(old_channels):
        label = old_channel.get("label")
        color = old_channel.get("color")
        wavelength_id = old_channel.get("wavelength_id")
        old_attributes = (
            f"Old attributes: {label=}, {wavelength_id=}, {color=}"
        )
        label = new_channels[ind]["label"]
        wavelength_id = new_channels[ind]["wavelength_id"]
        color = new_channels[ind]["color"]
        new_attributes = (
            f"New attributes: {label=}, {wavelength_id=}, {color=}"
        )
        logging.info(
            "Omero channel update:\n"
            f"    {old_attributes}\n"
            f"    {new_attributes}"
        )

    return new_channels

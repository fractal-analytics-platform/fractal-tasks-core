"""
Copyright 2022 (C)
    Friedrich Miescher Institute for Biomedical Research and
    University of Zurich

    Original authors:
    Tommaso Comparin <tommaso.comparin@exact-lab.it>

    This file is part of Fractal and was originally developed by eXact lab
    S.r.l.  <exact-lab.it> under contract with Liberali Lab from the Friedrich
    Miescher Institute for Biomedical Research and Pelkmans Lab from the
    University of Zurich.

Helper functions to address channels via OME-NGFF/OMERO metadata
"""
import logging
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import zarr
from pydantic import BaseModel

from fractal_tasks_core import __OME_NGFF_VERSION__


if __OME_NGFF_VERSION__ != "0.4":
    NotImplementedError(
        f"OME NGFF {__OME_NGFF_VERSION__} is not supported "
        "in `lib_channels.py`"
    )


class Window(BaseModel):
    """
    Custom class for Omero-channel window, based on OME-NGFF v0.4.
    """

    min: Optional[int]
    """Do not change. It will be set to ``0`` by default."""

    max: Optional[int]
    """Do not change. It will be set according to bit-depth of the images by
    default (e.g. 65535 for 16 bit images)."""

    start: int
    """Lower-bound rescaling value for visualization."""

    end: int
    """Upper-bound rescaling value for visualization."""


class OmeroChannel(BaseModel):
    """
    Custom class for Omero channels, based on OME-NGFF v0.4.
    """

    # Custom

    wavelength_id: str
    """Unique ID for the channel wavelength, e.g. ``A01_C01``."""

    index: Optional[int]
    """Do not change. For internal use only."""

    # From OME-NGFF v0.4 transitional metadata

    label: Optional[str]
    """Name of the channel"""

    window: Optional[Window]
    """Optional ``Window`` object to set default display settings for
    napari."""

    color: Optional[str]
    """Optional hex colormap to display the channel in napari
    (e.g. ``00FFFF``)."""

    active: bool = True
    """Should this channel be shown in the viewer?"""

    coefficient: int = 1
    """Do not change. Omero-channel attribute. """

    inverted: bool = False
    """Do not change. Omero-channel attribute."""


class ChannelNotFoundError(ValueError):
    """
    Custom error for when ``get_channel_from_list`` fails, that can be captured
    and handled upstream if needed.
    """

    pass


def check_unique_wavelength_ids(channels: List[OmeroChannel]):
    """
    Check that the `wavelength_id` attributes of a channel list are unique
    """
    wavelength_ids = [c.wavelength_id for c in channels]
    if len(set(wavelength_ids)) < len(wavelength_ids):
        raise ValueError(
            f"Non-unique wavelength_id's in {wavelength_ids}\n" f"{channels=}"
        )


def check_well_channel_labels(*, well_zarr_path: str) -> None:
    """
    Check that the channel labels for a well are unique

    First identify the channel-labels list for each image in the well, then
    compare lists and verify their intersection is empty

    :param well_zarr_path: path to an OME-NGFF well zarr group
    """

    # Iterate over all images (multiplexing cycles, multi-FOVs, ...)
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
    Extract a channel from OME-NGFF zarr attributes

    This is a helper function that combines ``get_omero_channel_list`` with
    ``get_channel_from_list``.

    :param image_zarr_path: Path to an OME-NGFF image zarr group
    :param label: ``label`` attribute of the channel to be extracted
    :param wavelength_id: ``wavelength_id`` attribute of the channel to be
                          extracted
    :returns: A single channel dictionary
    """
    omero_channels = get_omero_channel_list(image_zarr_path=image_zarr_path)
    channel = get_channel_from_list(
        channels=omero_channels, label=label, wavelength_id=wavelength_id
    )
    return channel


def get_omero_channel_list(*, image_zarr_path: str) -> List[OmeroChannel]:
    """
    Extract the list of channels from OME-NGFF zarr attributes

    :param image_zarr_path: Path to an OME-NGFF image zarr group
    :returns: A list of channel dictionaries
    """
    group = zarr.open_group(image_zarr_path, mode="r+")
    channels_dicts = group.attrs["omero"]["channels"]
    channels = [OmeroChannel(**c) for c in channels_dicts]
    return channels


def get_channel_from_list(
    *,
    channels: List[OmeroChannel],
    label: Optional[str] = None,
    wavelength_id: Optional[str] = None,
) -> OmeroChannel:
    """
    Find matching channel in a list

    Find the channel that has the required values of ``label`` and/or
    ``wavelength_id``, and identify its positional index (which also
    corresponds to its index in the zarr array).

    :param channels: A list of channel dictionary, where each channel includes
                     (at least) the ``label`` and ``wavelength_id`` keys
    :param label: The label to look for in the list of channels.
    :param wavelength_id: The wavelength_id to look for in the list of
                          channels.
    :returns: A single channel dictionary
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
    channels: List[OmeroChannel],
    bit_depth: int,
    label_prefix: Optional[str] = None,
) -> List[Dict[str, Union[str, int, bool, Dict[str, int]]]]:
    """
    Update a channel list to use it in the OMERO/channels metadata

    Given a list of channel dictionaries, update each one of them by:
        1. Adding a label (if missing);
        2. Adding a set of OMERO-specific attributes;
        3. Discarding all other attributes.

    The ``new_channels`` output can be used in the
    ``attrs["omero"]["channels"]`` attribute of an image group.

    :param channels: A list of channel dictionaries (each one must include the
                     ``wavelength_id`` key).
    :param bit_depth: bit depth
    :param label_prefix: TBD
    :returns: ``new_channels``, a new list of consistent channel dictionaries
              that can be written to OMERO metadata.

    """

    new_channels = [c.copy(deep=True) for c in channels]
    default_colors = ["00FFFF", "FF00FF", "FFFF00"]

    for channel in new_channels:
        wavelength_id = channel.wavelength_id

        # If channel.label is None, set it to a default value
        if channel.label is None:
            default_label = wavelength_id
            if label_prefix:
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

    # Check that channel labels are unique for this image
    labels = [c.label for c in new_channels]
    if len(set(labels)) < len(labels):
        raise ValueError(f"Non-unique labels in {new_channels=}")

    new_channels_dictionaries = [
        c.dict(exclude={"index"}, exclude_unset=True) for c in new_channels
    ]

    return new_channels_dictionaries

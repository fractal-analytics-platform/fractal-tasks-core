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

Handle OMERO-related metadata
"""
from typing import Any
from typing import Dict
from typing import List
from typing import Sequence


def define_omero_channels(
    actual_channels: Sequence[str],
    channel_parameters: Dict[str, Any],
    bit_depth: int,
) -> List[Dict[str, Any]]:
    """
    Prepare the .attrs["omero"]["channels"] attribute of an image group

    :param actual_channels: TBD
    :param channel_parameters: TBD
    :param bit_depth: TBD
    :returns: omero_channels
    """

    omero_channels = []
    default_colormaps = ["00FFFF", "FF00FF", "FFFF00"]
    for channel in actual_channels:

        # Set colormap. If missing, use the default ones (for the first three
        # channels) or gray
        colormap = channel_parameters[channel].get("colormap", None)
        if colormap is None:
            try:
                colormap = default_colormaps.pop()
            except IndexError:
                colormap = "808080"

        omero_channels.append(
            {
                "active": True,
                "coefficient": 1,
                "color": colormap,
                "family": "linear",
                "inverted": False,
                "label": channel_parameters[channel].get("label", channel),
                "window": {
                    "min": 0,
                    "max": 2**bit_depth - 1,
                },
            }
        )

        try:
            omero_channels[-1]["window"]["start"] = channel_parameters[
                channel
            ]["start"]
            omero_channels[-1]["window"]["end"] = channel_parameters[channel][
                "end"
            ]
        except KeyError:
            pass

    return omero_channels

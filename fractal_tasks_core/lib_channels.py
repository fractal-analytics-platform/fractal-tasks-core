from typing import Dict
from typing import Sequence


def get_channel(
    *, channels: Sequence[Dict], label: str = None, wavelength_id: str = None
):
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

    """
    if label:
        if wavelength_id:
            matching_channels = [
                c
                for c in channels
                if (
                    c["label"] == label and c["wavelength_id"] == wavelength_id
                )
            ]
        else:
            matching_channels = [c for c in channels if c["label"] == label]
    else:
        if wavelength_id:
            matching_channels = [
                c for c in channels if c["wavelength_id"] == wavelength_id
            ]
        else:
            raise ValueError(
                "get_channel requires at least one in {label,wavelength_id} "
                "arguments"
            )

    if len(matching_channels) > 1:
        raise ValueError(f"Inconsistent set of channels: {channels}")
    elif len(matching_channels) == 0:
        raise ValueError(
            f"No channel found in {channels} for {label=} and {wavelength_id=}"
        )

    channel = matching_channels[0]
    label = channel["label"]
    wavelength_id = channel["wavelength_id"]
    array_index = channels.index(channel)
    return label, wavelength_id, array_index

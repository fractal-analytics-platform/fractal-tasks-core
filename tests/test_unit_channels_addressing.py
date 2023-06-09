import json
from pathlib import Path

import pytest
from devtools import debug

from fractal_tasks_core.lib_channels import Channel
from fractal_tasks_core.lib_channels import get_channel_from_list


def test_get_channel_from_list(testdata_path: Path):

    # Read JSON data and cast into `Channel`s
    with (testdata_path / "omero/channels_list.json").open("r") as f:
        omero_channels_dict = json.load(f)
    omero_channels = [Channel(**c) for c in omero_channels_dict]
    debug(omero_channels)

    # Extract a channel from a list / case 1
    channel = get_channel_from_list(channels=omero_channels, label="label_1")
    debug(channel)
    assert channel.label == "label_1"
    assert channel.wavelength_id == "wavelength_id_1"
    assert channel.index == 0
    # Extract a channel from a list / case 2
    channel = get_channel_from_list(
        channels=omero_channels, wavelength_id="wavelength_id_2"
    )
    debug(channel)
    assert channel.label == "label_2"
    assert channel.wavelength_id == "wavelength_id_2"
    assert channel.index == 1
    # Extract a channel from a list / case 3
    channel = get_channel_from_list(
        channels=omero_channels,
        label="label_2",
        wavelength_id="wavelength_id_2",
    )
    debug(channel)
    assert channel.label == "label_2"
    assert channel.wavelength_id == "wavelength_id_2"
    assert channel.index == 1
    # Extract a channel from a list / case 4
    with pytest.raises(ValueError):
        channel = get_channel_from_list(channels=omero_channels)

import json
from pathlib import Path

from devtools import debug

from fractal_tasks_core.lib_channels import get_channel_from_list


def test_get_channel(testdata_path: Path):
    with (testdata_path / "omero/channels_list.json").open("r") as f:
        omero_channels = json.load(f)
    debug(omero_channels)

    channel = get_channel_from_list(channels=omero_channels, label="label_1")
    debug(channel)
    assert channel["label"] == "label_1"
    assert channel["wavelength_id"] == "wavelength_id_1"
    assert channel["index"] == 0

    channel = get_channel_from_list(
        channels=omero_channels, wavelength_id="wavelength_id_2"
    )
    debug(channel)
    assert channel["label"] == "label_2"
    assert channel["wavelength_id"] == "wavelength_id_2"
    assert channel["index"] == 1

    channel = get_channel_from_list(
        channels=omero_channels,
        label="label_2",
        wavelength_id="wavelength_id_2",
    )
    debug(channel)
    assert channel["label"] == "label_2"
    assert channel["wavelength_id"] == "wavelength_id_2"
    assert channel["index"] == 1

import json
from pathlib import Path

from devtools import debug

from fractal_tasks_core.lib_channels import get_channel


def test_get_channel(testdata_path: Path):
    with (testdata_path / "omero/channels_list.json").open("r") as f:
        omero_channels = json.load(f)
    debug(omero_channels)

    label, wl_id, index = get_channel(channels=omero_channels, label="label_1")
    assert wl_id == "wavelength_id_1"
    assert index == 0
    label, wl_id, index = get_channel(
        channels=omero_channels, wavelength_id="wavelength_id_2"
    )
    assert label == "label_2"
    assert index == 1

    label, wl_id, index = get_channel(
        channels=omero_channels,
        label="label_2",
        wavelength_id="wavelength_id_2",
    )
    assert label == "label_2"
    assert wl_id == "wavelength_id_2"
    assert index == 1

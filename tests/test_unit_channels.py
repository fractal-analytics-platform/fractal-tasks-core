import json
from pathlib import Path

import jsonschema
import pytest
from devtools import debug

from fractal_tasks_core.lib_channels import Channel
from fractal_tasks_core.lib_channels import define_omero_channels
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


@pytest.fixture(scope="session")
def omero_channel_schema():
    import urllib.request
    from fractal_tasks_core import __OME_NGFF_VERSION__

    url = (
        "https://raw.githubusercontent.com/ome/ngff/main/"
        f"{__OME_NGFF_VERSION__}/schemas/image.schema"
    )
    debug(url)
    with urllib.request.urlopen(url) as fin:
        full_schema = json.load(fin)
    yield full_schema["properties"]["omero"]["properties"]["channels"]["items"]


def test_define_omero_channels(testdata_path: Path, omero_channel_schema):
    """
    GIVEN a list of our custom `Channel` objects
    WHEN calling `define_omero_channels`
    THEN
        the output channel dictionaries are valid Omero channels according to
        the OME-NGFF schema
    """
    # Read JSON data and cast into `Channel`s
    with (testdata_path / "omero/channels_list.json").open("r") as f:
        omero_channels_dict = json.load(f)
    omero_channels = [Channel(**c) for c in omero_channels_dict]
    debug(omero_channels)

    # Call define_omero_channels
    processed_channels = define_omero_channels(
        channels=omero_channels, bit_depth=16, label_prefix="prefix"
    )
    debug(processed_channels)

    # Validate output of define_omero_channels
    for channel_dict in processed_channels:
        assert "color" in channel_dict
        assert "label" in channel_dict
        assert "index" not in channel_dict
        assert set(channel_dict["window"].keys()) == {
            "min",
            "max",
            "start",
            "end",
        }
        jsonschema.validate(instance=channel_dict, schema=omero_channel_schema)

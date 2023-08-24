import json
from pathlib import Path

import jsonschema
import pytest
import zarr
from devtools import debug

from fractal_tasks_core.lib_channels import check_unique_wavelength_ids
from fractal_tasks_core.lib_channels import check_well_channel_labels
from fractal_tasks_core.lib_channels import define_omero_channels
from fractal_tasks_core.lib_channels import get_channel_from_list
from fractal_tasks_core.lib_channels import OmeroChannel


def test_check_unique_wavelength_ids():
    # Fail for non-unique wavelength_id attributes
    with pytest.raises(ValueError) as e:
        check_unique_wavelength_ids(
            [
                OmeroChannel(wavelength_id="a"),
                OmeroChannel(wavelength_id="a"),
            ]
        )
    debug(e.value)


def test_check_well_channel_labels(tmp_path):
    well_group_path = str(tmp_path / "well.zarr")
    debug(well_group_path)
    well_group = zarr.group(well_group_path)
    well_group.attrs.put(
        dict(
            well=dict(
                images=[
                    dict(path="0"),
                    dict(path="1"),
                ]
            )
        )
    )
    debug(well_group.attrs.asdict())
    image0_group = well_group.create_group("0")
    image1_group = well_group.create_group("1")
    image0_group.attrs.put(
        dict(
            omero=dict(
                channels=[
                    OmeroChannel(
                        wavelength_id="id_1", label="non_unique_label"
                    ).dict(),
                ]
            )
        )
    )
    image1_group.attrs.put(
        dict(
            omero=dict(
                channels=[
                    OmeroChannel(
                        wavelength_id="id_1", label="non_unique_label"
                    ).dict(),
                ]
            )
        )
    )
    debug(image0_group.attrs)
    debug(image1_group.attrs)

    with pytest.raises(ValueError) as e:
        check_well_channel_labels(well_zarr_path=well_group_path)
    debug(e.value)
    assert "Non-unique" in str(e.value)


def test_get_channel_from_list(testdata_path: Path):

    # Read JSON data and cast into `OmeroChannel`s
    with (testdata_path / "omero/channels_list.json").open("r") as f:
        omero_channels_dict = json.load(f)
    omero_channels = [OmeroChannel(**c) for c in omero_channels_dict]
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
    with pytest.raises(ValueError) as e:
        channel = get_channel_from_list(channels=omero_channels)
    debug(e.value)

    # Extract a channel from an inconsistent list
    channels = [
        OmeroChannel(wavelength_id="id", label="label"),
        OmeroChannel(wavelength_id="id", label="label"),
    ]
    with pytest.raises(ValueError) as e:
        get_channel_from_list(channels=channels, label="label")
    debug(e.value)
    assert "Inconsistent" in str(e.value)


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
    GIVEN a list of our custom `OmeroChannel` objects
    WHEN calling `define_omero_channels`
    THEN
        the output channel dictionaries are valid Omero channels according to
        the OME-NGFF schema
    """
    # Read JSON data and cast into `OmeroChannel`s
    with (testdata_path / "omero/channels_list.json").open("r") as f:
        omero_channels_dict = json.load(f)
    omero_channels = [OmeroChannel(**c) for c in omero_channels_dict]
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

    # Fail, due to non-unique labels
    omero_channels = [
        OmeroChannel(wavelength_id="id", label="label"),
        OmeroChannel(wavelength_id="id", label="label"),
    ]
    debug(omero_channels)
    with pytest.raises(ValueError) as e:
        define_omero_channels(
            channels=omero_channels, bit_depth=16, label_prefix="prefix"
        )
    debug(e.value)
    assert "Non-unique" in str(e.value)


def test_color_validator():
    """
    Test the pydantic validator for hex colors in OmeroChannel.
    """

    valid_colors = ["aaaaaa", "abc123", "AAAAAA", "000000", "00ff00"]
    for c in valid_colors:
        OmeroChannel(wavelength_id="A01_C01", color=c)

    invalid_colors = ["#aaaaaa", "abc", "abc12", "abcde-"]
    for c in invalid_colors:
        with pytest.raises(ValueError):
            OmeroChannel(wavelength_id="A01_C01", color=c)

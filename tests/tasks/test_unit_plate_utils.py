"""Unit tests for fractal_tasks_core._plate_utils."""

import pytest

from fractal_tasks_core._utils import (
    HCSZarrUrl,
    _parse_hcs_zarr_url,
    group_by_plate,
    group_by_well,
    split_well_path_image_path,
)


def test_group_by_plate_single_plate():
    urls = ["/base/plate.zarr/A/01/0", "/base/plate.zarr/A/01/1"]
    result = group_by_plate(urls)
    assert list(result.keys()) == ["/base/plate.zarr"]
    assert len(result["/base/plate.zarr"]) == 2


def test_group_by_plate_multiple_plates():
    urls = ["/base/plate1.zarr/A/01/0", "/base/plate2.zarr/A/01/0"]
    result = group_by_plate(urls)
    assert len(result) == 2
    assert "/base/plate1.zarr" in result
    assert "/base/plate2.zarr" in result


def test_group_by_well_groups_by_well():
    urls = [
        "/base/plate.zarr/A/01/0",
        "/base/plate.zarr/A/01/1",
        "/base/plate.zarr/B/02/0",
    ]
    result = group_by_well(urls)
    assert len(result) == 2
    assert len(result["/base/plate.zarr/A/01"]) == 2
    assert len(result["/base/plate.zarr/B/02"]) == 1


def test_parse_hcs_zarr_url_too_short_raises():
    with pytest.raises(ValueError, match="too short"):
        _parse_hcs_zarr_url(["plate/row/col"])


def test_split_well_path_too_short_raises():
    with pytest.raises(ValueError, match="too short"):
        split_well_path_image_path("plate/row/col")


def test_split_well_path_returns_correct_parts():
    well_path, image_path = split_well_path_image_path("/base/plate.zarr/A/01/0")
    assert well_path == "/base/plate.zarr/A/01"
    assert image_path == "0"


def test_hcs_zarr_url_properties():
    url = HCSZarrUrl(
        base="/base", plate="plate.zarr", row="A", column="01", image_path="0"
    )
    assert url.plate_url == "/base/plate.zarr"
    assert url.well_url == "/base/plate.zarr/A/01"
    assert url.zarr_url == "/base/plate.zarr/A/01/0"

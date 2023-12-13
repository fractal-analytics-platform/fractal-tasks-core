import pytest
import zarr
from devtools import debug

from fractal_tasks_core.ome_zarr.utils import (
    _find_omengff_acquisition,
)
from fractal_tasks_core.ome_zarr.utils import (
    get_parameters_from_metadata,
)


def test_find_omengff_acquisition(tmp_path):
    """
    Fail because the parent zarr group (which is expected to be a ome-zarr
    "well" group) does not have a .zattrs file.
    """
    image_name = "my_image"
    well_zarr_path = tmp_path / "well.zarr"
    well_group = zarr.group(well_zarr_path)
    debug(well_zarr_path)
    debug(well_group)
    with pytest.raises(ValueError) as e:
        image_zarr_path = well_zarr_path / image_name
        _find_omengff_acquisition(image_zarr_path)
    debug(e.value)


def test_get_parameters_from_metadata(tmp_path):

    ACQUISITION = 99
    IMAGE_NAME = "cycle_123"
    COARSENING_XY = 2
    NUM_LEVELS = 5
    IMAGE_EXTENSION = "png"

    well_zarr_path = tmp_path / "well.zarr"
    well_group = zarr.group(well_zarr_path)
    well_group.attrs.put(
        dict(
            well=dict(
                images=[
                    dict(acquisition=ACQUISITION, path=IMAGE_NAME),
                ]
            )
        )
    )
    debug(well_group.attrs.asdict())
    metadata = dict(
        coarsening_xy=COARSENING_XY,
        num_levels=NUM_LEVELS,
        image_extension=IMAGE_EXTENSION,
    )
    parameters = get_parameters_from_metadata(
        keys=[
            "num_levels",
            "coarsening_xy",
            "image_extension",
        ],
        metadata=metadata,
        image_zarr_path=(well_zarr_path / IMAGE_NAME),
    )
    debug(parameters)
    assert parameters["acquisition"] == ACQUISITION
    assert parameters["num_levels"] == NUM_LEVELS
    assert parameters["coarsening_xy"] == COARSENING_XY
    assert parameters["image_extension"] == IMAGE_EXTENSION

    # Fail due to missing key
    with pytest.raises(KeyError) as e:
        get_parameters_from_metadata(
            keys=["invalid_key"],
            metadata=metadata,
            image_zarr_path=(well_zarr_path / IMAGE_NAME),
        )
    debug(e.value)

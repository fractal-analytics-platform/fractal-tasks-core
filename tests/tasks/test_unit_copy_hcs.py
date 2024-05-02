"""
Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
University of Zurich

Original authors:
Marco Franzon <marco.franzon@exact-lab.it>
Tommaso Comparin <tommaso.comparin@exact-lab.it>

This file is part of Fractal and was originally developed by eXact lab S.r.l.
<exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
Institute for Biomedical Research and Pelkmans Lab from the University of
Zurich.
"""
import shutil
from pathlib import Path

import pytest
import zarr
from devtools import debug

from ._validation import validate_schema
from fractal_tasks_core.tasks.copy_ome_zarr_hcs_plate import (
    copy_ome_zarr_hcs_plate,
)
from fractal_tasks_core.zarr_utils import OverwriteNotAllowedError

expected_MIP_plate_attrs = {
    "plate": {
        "acquisitions": [
            {"id": 0, "name": "20200812-CardiomyocyteDifferentiation14-Cycle1"}
        ],
        "columns": [{"name": "03"}],
        "rows": [{"name": "B"}],
        "wells": [{"columnIndex": 0, "path": "B/03", "rowIndex": 0}],
    }
}


def test_MIP(
    tmp_path: Path,
    zenodo_zarr: list[str],
):

    # Init
    zarr_path = tmp_path / "tmp_out/"

    # Load zarr array from zenodo
    zenodo_zarr_3D, zenodo_zarr_2D = zenodo_zarr[:]
    shutil.copytree(zenodo_zarr_3D, str(zarr_path / Path(zenodo_zarr_3D).name))

    zarr_urls = []
    zarr_dir = "/".join(zenodo_zarr_3D.split("/")[:-1])
    zarr_urls = [Path(zarr_dir, "plate.zarr/B/03/0").as_posix()]

    parallelization_list = copy_ome_zarr_hcs_plate(
        zarr_urls=zarr_urls,
        zarr_dir="tmp_out",
        overwrite=True,
    )["parallelization_list"]
    debug(parallelization_list)

    # Run again, with overwrite=True
    parallelization_list_2 = copy_ome_zarr_hcs_plate(
        zarr_urls=zarr_urls,
        zarr_dir="tmp_out",
        overwrite=True,
    )["parallelization_list"]
    assert parallelization_list_2 == parallelization_list

    # Run again, with overwrite=False
    with pytest.raises(OverwriteNotAllowedError):
        _ = copy_ome_zarr_hcs_plate(
            zarr_urls=zarr_urls,
            zarr_dir="tmp_out",
            overwrite=False,
        )

    # OME-NGFF JSON validation for plate & well
    image_zarr = Path(parallelization_list[0]["zarr_url"])
    well_zarr = image_zarr.parent
    plate_zarr = image_zarr.parents[2]
    validate_schema(path=str(well_zarr), type="well")
    validate_schema(path=str(plate_zarr), type="plate")

    # Validate plate zarr attributes:
    plate_attrs = zarr.open_group(plate_zarr).attrs.asdict()
    assert plate_attrs == expected_MIP_plate_attrs

import glob
import json
import logging
import urllib
from pathlib import Path
from typing import Dict

import anndata as ad
import dask.array as da
import numpy as np
import zarr
from devtools import debug
from jsonschema import validate

from fractal_tasks_core import __OME_NGFF_VERSION__
from fractal_tasks_core.lib_regions_of_interest import (
    convert_ROI_table_to_indices,
)
from fractal_tasks_core.lib_zattrs_utils import extract_zyx_pixel_sizes


def validate_schema(*, path: str, type: str):
    url: str = (
        "https://raw.githubusercontent.com/ome/ngff/main/"
        f"{__OME_NGFF_VERSION__}/schemas/{type}.schema"
    )
    debug(url)
    with urllib.request.urlopen(url) as fin:
        schema: Dict = json.load(fin)
    debug(path)
    debug(type)
    with open(f"{path}/.zattrs", "r") as fin:
        zattrs = json.load(fin)
    validate(instance=zattrs, schema=schema)


def check_file_number(*, zarr_path: Path):
    """
    Example input:
        zarr_path = Path("/SOME/PATH/plate.zarr/row/col/fov/")

    Relevant glob for zarr_path
        zarr_path / 0 / c / z / y / x

    """
    chunkfiles_on_disk = glob.glob(str(zarr_path / "0/*/*/*/*"))
    debug(chunkfiles_on_disk)
    num_chunkfiles_on_disk = len(chunkfiles_on_disk)

    zarr_chunks = da.from_zarr(str(zarr_path / "0/")).chunks
    debug(zarr_chunks)
    num_chunkfiles_from_zarr = 1
    for c in zarr_chunks:
        num_chunkfiles_from_zarr *= len(c)

    assert num_chunkfiles_from_zarr == num_chunkfiles_on_disk


def validate_axes_and_coordinateTransformations(image_zarr: Path):
    """
    Check that the length of a "scale" transformation matches with the number
    of axes
    """
    zattrs_file = image_zarr / ".zattrs"
    with zattrs_file.open("r") as f:
        zattrs = json.load(f)
    debug(zattrs)
    multiscale = zattrs["multiscales"][0]
    dataset = multiscale["datasets"][0]
    axes = multiscale["axes"]
    debug(axes)
    for transformation in dataset["coordinateTransformations"]:
        debug(transformation)
        if transformation["type"] != "scale":
            continue
        else:
            assert len(transformation["scale"]) == len(axes)
    raise


def validate_labels_and_measurements(
    image_zarr: Path, *, label_name: str, table_name: str
):

    label_path = str(image_zarr / "labels" / label_name / "0")
    table_path = str(image_zarr / "tables" / table_name)
    debug(label_path)
    debug(table_path)

    # Load label array
    labels = da.from_zarr(label_path).compute()
    list_label_values = list(np.unique(labels))

    # Create list of FOV-ROI indices
    zattrs_file = str(image_zarr / ".zattrs")
    FOV_table_path = str(image_zarr / "tables/FOV_ROI_table")
    ROI_table = ad.read_zarr(FOV_table_path)
    full_res_pxl_sizes_zyx = extract_zyx_pixel_sizes(zattrs_file, level=0)
    list_indices = convert_ROI_table_to_indices(
        ROI_table,
        level=0,
        coarsening_xy=2,
        full_res_pxl_sizes_zyx=full_res_pxl_sizes_zyx,
    )
    num_ROIs = len(list_indices)
    debug(list_indices)

    # Check that different ROIs do not share labels
    for ind_ROI_1 in range(num_ROIs):
        s_z, e_z, s_y, e_y, s_x, e_x = list_indices[ind_ROI_1]
        region_1 = (slice(s_z, e_z), slice(s_y, e_y), slice(s_x, e_x))
        for ind_ROI_2 in range(ind_ROI_1):
            s_z, e_z, s_y, e_y, s_x, e_x = list_indices[ind_ROI_2]
            region_2 = (slice(s_z, e_z), slice(s_y, e_y), slice(s_x, e_x))
            # Compare sets of unique labels for different ROIs
            labels_ROI_1 = set(np.unique(labels[region_1]))
            if 0 in labels_ROI_1:
                labels_ROI_1.remove(0)
            labels_ROI_2 = set(np.unique(labels[region_2]))
            if 0 in labels_ROI_2:
                labels_ROI_2.remove(0)
            intersection = labels_ROI_1 & labels_ROI_2
            assert not intersection

    # Load measurements
    try:
        table = ad.read_zarr(table_path)
    except zarr.errors.PathNotFoundError:
        logging.warning(
            f"{table_path} missing, skip validation of dataframe and of "
            "dataframe/label match"
        )
        return
    debug(table)
    if len(table) == 0:
        logging.warning(
            f"Table in {table_path} is empty, skip validation of "
            "dataframe and of dataframe/label match"
        )
        return

    list_table_label_values = [int(x) for x in list(table.obs["label"])]

    # Check that measurement labels are unique
    assert len(set(list_table_label_values)) == len(list_table_label_values)

    # Check match of label array/measurement (after removing the no-label 0
    # value from array)
    if list_label_values[0] == 0:
        list_label_values = list_label_values[1:]
    assert list_table_label_values == list_label_values

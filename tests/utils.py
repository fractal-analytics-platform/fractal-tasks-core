import glob
import json
import urllib
from pathlib import Path
from typing import Dict

import anndata as ad
import dask.array as da
import zarr
from devtools import debug
from jsonschema import validate

from fractal_tasks_core import __OME_NGFF_VERSION__


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


def validate_labels_and_measurements(
    image_zarr: Path, *, label_name: str, table_name: str
):

    # FIXME: clean up this test and make asserts as strict as possible

    label_path = str(image_zarr / "labels" / label_name / "0")
    table_path = str(image_zarr / "tables" / table_name)
    debug(label_path)
    debug(table_path)

    # Load label array
    labels = da.from_zarr(label_path)
    list_label_values = list(da.unique(labels).compute())

    # Check that labels are unique
    assert len(set(list_label_values)) == len(list_label_values)

    # Load measurements
    try:
        table = ad.read_zarr(table_path)
        list_table_label_values = [int(x) for x in list(table.obs["label"])]
    except zarr.errors.PathNotFoundError:
        print(
            f"{table_path} missing, skip validation of dataframe and of "
            "dataframe/label match"
        )
        return

    # Check that measurement labels are unique
    assert len(set(list_table_label_values)) == len(list_table_label_values)

    # Check match of label array/measurement (after removing the no-label 0
    # value from array)
    if list_label_values[0] == 0:
        list_label_values = list_label_values[1:]
    assert list_table_label_values == list_label_values

import glob
import json
import urllib
from pathlib import Path
from typing import Dict

import anndata as ad
import dask.array as da
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
    # FIXME: move this test at the end of a napari-workflow task

    label_path = str(image_zarr / "labels" / label_name / "0")
    table_path = str(image_zarr / "tables" / table_name)
    labels = da.from_zarr(label_path)
    list_label_values = list(da.unique(labels).compute())
    assert list_label_values[0] == 0
    list_label_values = list_label_values[1:]

    table = ad.read_zarr(table_path)
    list_table_label_values = [int(x) for x in list(table.obs["label"])]

    # Check that labels are unique in measurement dataframe
    assert len(set(list_table_label_values)) == len(list_table_label_values)

    # Check that labels are the same in measurement dataframe and labels array
    assert list_table_label_values == list_label_values

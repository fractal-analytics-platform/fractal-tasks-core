import dask
import dask.array as da
import zarr
from devtools import debug


def test_dask_array_to_zarr(tmp_path):
    """
    This test fails with dask=2022.7, when upgrading from zarr=2.13.3 to
    zarr=2.13.6. See https://github.com/dask/dask/issues/9841.
    """
    debug(f"{zarr.__version__=}")
    debug(f"{dask.__version__=}")
    zarrurl = str(tmp_path / "ones.zarr")
    debug(zarrurl)
    x = da.ones((1, 1))
    x.to_zarr(zarrurl, dimension_separator="/")


def test_dask_core_get_mapper(tmp_path):
    """
    This test verifies that we have a method to create a zarr store via fsspec,
    in different ways depending on the dask version (changed in
    https://github.com/dask/dask/pull/9790, which entered dask=2023.1.0).
    """
    debug(f"{zarr.__version__=}")
    debug(f"{dask.__version__=}")
    zarrurl = str(tmp_path / "ones.zarr")
    debug(zarrurl)
    if dask.__version__ < "2023":
        store = da.core.get_mapper(zarrurl)
    else:
        store = zarr.storage.FSStore(zarrurl, dimension_separator="/")
    debug(store)

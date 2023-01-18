import dask
import dask.array as da
import zarr
from devtools import debug


def test_dask_array_to_zarr(tmp_path):
    """
    This test fails with dask=2022.7, when upgrading from zarr=2.13.3 to
    zarr=2.13.6. See https://github.com/dask/dask/issues/9841.
    """

    print(f"{zarr.__version__=}")
    print(f"{dask.__version__=}")

    zarrurl = str(tmp_path / "ones.zarr")

    x = da.ones((1, 1))
    x.to_zarr(zarrurl, dimension_separator="/")


def test_dask_core_get_mapper():
    """
    Since we use `da.core.get_mapper()` (from fsspec) in the tasks, this test
    checks that it is available.
    """
    print(f"{zarr.__version__=}")
    print(f"{dask.__version__=}")
    dummy = da.core.get_mapper()
    debug(dummy)

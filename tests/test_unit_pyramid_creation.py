import dask.array as da
import pytest
import zarr
from devtools import debug

from fractal_tasks_core.pyramids import build_pyramid


def test_build_pyramid(tmp_path):

    # Fail because only 2D,3D,4D are supported / A
    zarrurl = str(tmp_path / "A.zarr")
    # Specify the dimension separator as '/'
    store = zarr.DirectoryStore(f"{zarrurl}/0", dimension_separator="/")
    # Save the Dask array to the Zarr store
    da.ones(shape=(16,)).to_zarr(store)
    with pytest.raises(ValueError) as e:
        build_pyramid(zarrurl=zarrurl)
    debug(e.value)
    assert "ndims" in str(e.value)

    # Fail because only 2D,3D,4D are supported / B
    zarrurl = str(tmp_path / "B.zarr")
    # Specify the dimension separator as '/'
    store = zarr.DirectoryStore(f"{zarrurl}/0", dimension_separator="/")
    da.ones(shape=(2, 2, 2, 2, 2)).to_zarr(store)
    with pytest.raises(ValueError) as e:
        build_pyramid(zarrurl=zarrurl)
    debug(e.value)
    assert "ndims" in str(e.value)

    # Fail because there is not enough data for coarsening
    zarrurl = str(tmp_path / "C.zarr")
    # Specify the dimension separator as '/'
    store = zarr.DirectoryStore(f"{zarrurl}/0", dimension_separator="/")
    da.ones(shape=(4, 4)).to_zarr(store)
    with pytest.raises(ValueError) as e:
        build_pyramid(zarrurl=zarrurl, coarsening_xy=10)
    debug(e.value)
    assert "but previous level has shape" in str(e.value)

    # Succeed
    zarrurl = str(tmp_path / "D.zarr")
    # Specify the dimension separator as '/'
    store = zarr.DirectoryStore(f"{zarrurl}/0", dimension_separator="/")
    da.ones(shape=(8, 8)).to_zarr(store)
    build_pyramid(zarrurl=zarrurl, coarsening_xy=2, num_levels=3)
    level_1 = da.from_zarr(f"{zarrurl}/1")
    level_2 = da.from_zarr(f"{zarrurl}/2")
    debug(level_1)
    debug(level_2)
    assert level_1.shape == (4, 4)
    assert level_2.shape == (2, 2)

    # Succeed
    zarrurl = str(tmp_path / "E.zarr")
    # Specify the dimension separator as '/'
    store = zarr.DirectoryStore(f"{zarrurl}/0", dimension_separator="/")
    da.ones(shape=(243 + 2, 243)).to_zarr(store)
    build_pyramid(zarrurl=zarrurl, coarsening_xy=3, num_levels=6, chunksize=9)
    level_1 = da.from_zarr(f"{zarrurl}/1")
    level_2 = da.from_zarr(f"{zarrurl}/2")
    level_3 = da.from_zarr(f"{zarrurl}/3")
    level_4 = da.from_zarr(f"{zarrurl}/4")
    level_5 = da.from_zarr(f"{zarrurl}/5")
    debug(level_1)
    debug(level_2)
    debug(level_3)
    debug(level_4)
    debug(level_5)
    assert level_1.shape == (81, 81)
    assert level_1.chunksize == (9, 9)
    assert level_2.shape == (27, 27)
    assert level_2.chunksize == (9, 9)
    assert level_3.shape == (9, 9)
    assert level_3.chunksize == (9, 9)
    assert level_4.shape == (3, 3)
    assert level_5.shape == (1, 1)

    # check that open_array_kwargs has an effect
    zarrurl = tmp_path / "F.zarr"
    # Specify the dimension separator as '/'
    store = zarr.DirectoryStore(f"{zarrurl}/0", dimension_separator="/")
    da.ones(shape=(8, 8)).to_zarr(store)
    build_pyramid(
        zarrurl=zarrurl,
        coarsening_xy=2,
        num_levels=3,
        open_array_kwargs={"write_empty_chunks": True},
    )
    # check that empty chunks are written to disk
    assert (zarrurl / "1/0/0").exists()
    assert (zarrurl / "2/0/0").exists()

    zarrurl = tmp_path / "G.zarr"
    # Specify the dimension separator as '/'
    store = zarr.DirectoryStore(f"{zarrurl}/0", dimension_separator="/")
    da.zeros(shape=(8, 8)).to_zarr(store)
    build_pyramid(
        zarrurl=zarrurl,
        coarsening_xy=2,
        num_levels=3,
        open_array_kwargs={"write_empty_chunks": False, "fill_value": 0},
    )
    # check that empty chunks are not written to disk
    assert not (zarrurl / "1/0/0").exists()
    assert not (zarrurl / "2/0/0").exists()


def test_build_pyramid_overwrite(tmp_path):
    # Succeed
    zarrurl = str(tmp_path / "K.zarr")
    # Specify the dimension separator as '/'
    store = zarr.DirectoryStore(f"{zarrurl}/0", dimension_separator="/")
    da.ones(shape=(8, 8)).to_zarr(store)
    build_pyramid(zarrurl=zarrurl, coarsening_xy=2, num_levels=3)
    # Should fail because overwrite is not set
    with pytest.raises(ValueError):
        build_pyramid(
            zarrurl=zarrurl, coarsening_xy=2, num_levels=3, overwrite=False
        )
    # Should work
    build_pyramid(
        zarrurl=zarrurl, coarsening_xy=2, num_levels=3, overwrite=True
    )

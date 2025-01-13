import dask.array as da
import pytest
from devtools import debug

from fractal_tasks_core.pyramids import build_pyramid


def test_build_pyramid(tmp_path):

    # Fail because only 2D,3D,4D are supported / A
    zarrurl = str(tmp_path / "A.zarr")
    da.ones(shape=(16,)).to_zarr(f"{zarrurl}/0")
    with pytest.raises(ValueError) as e:
        build_pyramid(zarrurl=zarrurl)
    debug(e.value)
    assert "ndims" in str(e.value)

    # Fail because only 2D,3D,4D are supported / B
    zarrurl = str(tmp_path / "B.zarr")
    da.ones(shape=(2, 2, 2, 2, 2)).to_zarr(f"{zarrurl}/0")
    with pytest.raises(ValueError) as e:
        build_pyramid(zarrurl=zarrurl)
    debug(e.value)
    assert "ndims" in str(e.value)

    # Fail because there is not enough data for coarsening
    zarrurl = str(tmp_path / "C.zarr")
    da.ones(shape=(4, 4)).to_zarr(f"{zarrurl}/0")
    with pytest.raises(ValueError) as e:
        build_pyramid(zarrurl=zarrurl, coarsening_xy=10)
    debug(e.value)
    assert "but previous level has shape" in str(e.value)

    # Succeed
    zarrurl = str(tmp_path / "D.zarr")
    da.ones(shape=(8, 8)).to_zarr(f"{zarrurl}/0")
    build_pyramid(zarrurl=zarrurl, coarsening_xy=2, num_levels=3)
    level_1 = da.from_zarr(f"{zarrurl}/1")
    level_2 = da.from_zarr(f"{zarrurl}/2")
    debug(level_1)
    debug(level_2)
    assert level_1.shape == (4, 4)
    assert level_2.shape == (2, 2)

    # Succeed
    zarrurl = str(tmp_path / "E.zarr")
    da.ones(shape=(243 + 2, 243)).to_zarr(f"{zarrurl}/0")
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

    # Succeed
    zarrurl = str(tmp_path / "F.zarr")
    da.zeros(shape=(8, 8)).to_zarr(f"{zarrurl}/0")
    build_pyramid(
        zarrurl=zarrurl,
        coarsening_xy=2,
        num_levels=3,
        open_array_kwargs={"write_empty_chunks": False, "fill_value": 0},
    )
    level_1 = da.from_zarr(f"{zarrurl}/1")
    level_2 = da.from_zarr(f"{zarrurl}/2")
    assert level_1.shape == (4, 4)
    assert level_2.shape == (2, 2)
    # check that the empty chunks are not written to disk
    assert not (tmp_path / "F.zarr/1/0.0").exists()
    assert not (tmp_path / "F.zarr/2/0.0").exists()

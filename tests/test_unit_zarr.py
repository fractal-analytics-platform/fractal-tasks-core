import pytest
from devtools import debug

from fractal_tasks_core.lib_zarr import open_zarr_group_with_overwrite
from fractal_tasks_core.lib_zarr import OverwriteNotAllowedError


def test_open_zarr_group_with_overwrite(tmp_path):
    """
    Test open_zarr_group_with_overwrite

    See
    https://github.com/fractal-analytics-platform/fractal-tasks-core/issues/485
    """

    path_a = str(tmp_path / "group_a.zarr")
    path_b = str(tmp_path / "group_b.zarr")

    # If `overwrite=False` and the group does not exist, create it
    group_a = open_zarr_group_with_overwrite(path_a, overwrite=False)
    group_a.create_group("a1")
    group_a.create_group("a2")
    debug(group_a.info)
    assert set(group_a.group_keys()) == {"a1", "a2"}
    print()

    # If `overwrite=True` and the group does not exist, create it
    group_b = open_zarr_group_with_overwrite(path_b, overwrite=True)
    group_b.create_group("b1")
    group_b.create_group("b2")
    debug(group_b.info)
    assert set(group_b.group_keys()) == {"b1", "b2"}
    print()

    # If `overwrite=True` and the group already exists, replace it with a new
    # empty group
    group_a_new = open_zarr_group_with_overwrite(path_a, overwrite=True)
    debug(group_a_new.info)
    assert set(group_a_new.group_keys()) == set()
    print()

    # If `overwrite=False` and the group already exists, fail with a relevant
    # error
    with pytest.raises(OverwriteNotAllowedError) as e:
        open_zarr_group_with_overwrite(path_b, overwrite=False)
    print(e.value)

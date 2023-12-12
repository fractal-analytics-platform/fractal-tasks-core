# Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Tommaso Comparin <tommaso.comparin@exact-lab.it>
#
# This file is part of Fractal and was originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
"""
Subpackage with functions and classes related to table specifications (see
https://fractal-analytics-platform.github.io/fractal-tasks-core/tables).
"""
from typing import Any
from typing import Optional

import anndata as ad
import zarr.hierarchy

from .v1 import _write_table_v1
from fractal_tasks_core import __FRACTAL_TABLE_VERSION__


def write_table(
    image_group: zarr.hierarchy.Group,
    table_name: str,
    table: ad.AnnData,
    overwrite: bool = False,
    table_type: Optional[str] = None,
    table_attrs: Optional[dict[str, Any]] = None,
) -> zarr.group:
    """
    Write a table to a Zarr group.

    This is the general interface that should allow for a smooth coexistence of
    tables with different `fractal_table_version` values. Currently only V1 is
    defined and implemented. The assumption is that V2 should only change:

    1. The lower-level writing function (that is, `_write_table_v2`).
    2. The type of the table (which would also reflect into a more general type
        hint for `table`, in the current funciton);
    3. A different definition of what values of `table_attrs` are valid or
       invalid, to be implemented in `_write_table_v2`.
    4. Possibly, additional parameters for `_write_table_v2`, which will be
       optional parameters of `write_table` (so that `write_table` remains
       valid for both V1 and V2).

    Args:
        image_group:
            The image Zarr group where the table will be written.
        table_name:
            The name of the table.
        table:
            The table object (currently an AnnData object, for V1).
        overwrite:
            If `False`, check that the new table does not exist (either as a
            zarr sub-group or as part of the zarr-group attributes). In all
            cases, propagate parameter to low-level functions, to determine the
            behavior in case of an existing sub-group named as in `table_name`.
        table_type: `type` attribute for the table; in case `type` is also
            present in `table_attrs`, this function argument takes priority.
        table_attrs:
            If set, overwrite table_group attributes with table_attrs key/value
            pairs. If `table_type` is not provided, then `table_attrs` must
            include the `type` key.

    Returns:
        Zarr group of the table.
    """
    # Choose which version to use, giving priority to a value that is present
    # in table_attrs
    version = __FRACTAL_TABLE_VERSION__
    if table_attrs is not None:
        try:
            version = table_attrs["fractal_table_version"]
        except KeyError:
            pass

    if version == "1":
        return _write_table_v1(
            image_group,
            table_name,
            table,
            overwrite,
            table_type,
            table_attrs,
        )
    else:
        raise NotImplementedError(
            f"fractal_table_version='{version}' is not supported"
        )

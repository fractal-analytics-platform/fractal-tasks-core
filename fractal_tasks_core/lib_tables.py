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
FIXME
"""
import logging
from typing import Any
from typing import Optional

import anndata as ad
import zarr.hierarchy

from .lib_write import _write_elem_with_overwrite
from .lib_write import OverwriteNotAllowedError
from fractal_tasks_core import __FRACTAL_TABLE_VERSION__


logger = logging.getLogger(__name__)


def _write_table_v1(
    image_group: zarr.hierarchy.Group,
    table_name: str,
    table: ad.AnnData,
    overwrite: bool = False,
    table_attrs: Optional[dict[str, Any]] = None,
) -> zarr.group:
    """
    Handle multiple options for writing an AnnData table to a zarr group.

    1. Create the `tables` group, if needed.
    2. If `overwrite=False`, check that the new table does not exist (either in
       zarr attributes or as a zarr sub-group).
    3. Call the `_write_elem_with_overwrite` wrapper with the appropriate
       `overwrite` parameter.
    4. Update the `tables` attribute of the image group.
    5. If `table_attrs` is set, include this set of attributes in the
       new-table zarr group. One intended usage, within fractal-tasks-core, is
       to comply with a proposed change to the OME-NGFF specs
       (https://github.com/ome/ngff/pull/64).

    Args:
        image_group:
            The group to write to.
        table_name:
            The name of the new table.
        table:
            The AnnData table to write.
        overwrite:
            If `False`, check that the new table does not exist (either as a
            zarr sub-group or as part of the zarr-group attributes). In all
            cases, propagate parameter to `_write_elem_with_overwrite`, to
            determine the behavior in case of an existing sub-group named as
            `table_name`.
        table_attrs:
            If set, overwrite table_group attributes with table_attrs key/value
            pairs.

    Returns:
        Zarr group of the new table.
    """

    # Create tables group (if needed) and extract current_tables
    if "tables" not in set(image_group.group_keys()):
        tables_group = image_group.create_group("tables", overwrite=False)
    else:
        tables_group = image_group["tables"]
    current_tables = tables_group.attrs.asdict().get("tables", [])

    # If overwrite=False, check that the new table does not exist (either as a
    # zarr sub-group or as part of the zarr-group attributes)
    if not overwrite:
        if table_name in set(tables_group.group_keys()):
            error_msg = (
                f"Sub-group '{table_name}' of group {image_group.store.path} "
                f"already exists, but `{overwrite=}`.\n"
                "Hint: try setting `overwrite=True`."
            )
            logger.error(error_msg)
            raise OverwriteNotAllowedError(error_msg)
        if table_name in current_tables:
            error_msg = (
                f"Item '{table_name}' already exists in `tables` attribute of "
                f"group {image_group.store.path}, but `{overwrite=}`.\n"
                "Hint: try setting `overwrite=True`."
            )
            logger.error(error_msg)
            raise OverwriteNotAllowedError(error_msg)

    # If it's all OK, proceed and write the table
    _write_elem_with_overwrite(
        tables_group,
        table_name,
        table,
        overwrite=overwrite,
    )
    table_group = tables_group[table_name]

    # Update the `tables` metadata of the image group, if needed
    if table_name not in current_tables:
        new_tables = current_tables + [table_name]
        tables_group.attrs["tables"] = new_tables

    # Always add information about the fractal-roi-table version
    if table_attrs is None:
        table_attrs = dict(fractal_table_version=__FRACTAL_TABLE_VERSION__)
    elif table_attrs.get("fractal_table_version", None) is None:
        table_attrs["fractal_table_version"] = __FRACTAL_TABLE_VERSION__

    # Raise warning for non-compliance with table specs
    table_type = table_attrs.get("type", None)
    if table_type is None:
        pass
    elif table_type == "roi_table":
        pass
    elif table_type == "masking_roi_table":
        try:
            table_attrs["region"]["path"]
            table_attrs["instance_key"]
        except KeyError as e:
            logger.warning(
                f"Current `masking_roi_table` does not comply with Fractal "
                f"table specs V1. Original error: KeyError: {str(e)}"
            )
    elif table_type == "feature_table":
        try:
            table_attrs["region"]["path"]
            table_attrs["instance_key"]
        except KeyError as e:
            logger.warning(
                f"Current `masking_roi_table` does not comply with Fractal "
                f"table specs V1. Original error: KeyError: {str(e)}"
            )
    else:
        logger.warning(f"Unknown table type `{table_type}`.")

    # Update table_group attributes with table_attrs key/value pairs
    table_group.attrs.update(**table_attrs)

    return table_group


def write_table(
    image_group: zarr.hierarchy.Group,
    table_name: str,
    table: ad.AnnData,
    overwrite: bool = False,
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
            cases, propagate parameter to `_write_elem_with_overwrite`, to
            determine the behavior in case of an existing sub-group named as
            `table_name`.
        table_attrs:
            If set, overwrite table_group attributes with table_attrs key/value
            pairs.

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
            table_attrs,
        )
    else:
        raise NotImplementedError(
            f"fractal_table_version {version} is not supported"
        )
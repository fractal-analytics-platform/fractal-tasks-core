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
Module with custom wrappers of the Zarr API.
"""
import logging
from collections.abc import MutableMapping
from typing import Any
from typing import Optional
from typing import Union

import anndata as ad
import zarr
from anndata.experimental import write_elem
from zarr.errors import ContainsGroupError


class OverwriteNotAllowedError(RuntimeError):
    pass


def open_zarr_group_with_overwrite(
    path: Union[str, MutableMapping],
    *,
    overwrite: bool,
    logger: Optional[logging.Logger] = None,
    **open_group_kwargs: Any,
) -> zarr.Group:
    """
    Wrap `zarr.open_group` and add `overwrite` argument.

    This wrapper sets `mode="w"` for `overwrite=True` and `mode="w-"` for
    `overwrite=False`.

    The expected behavior is


    * if the group does not exist, create it (independently on `overwrite`);
    * if the group already exists and `overwrite=True`, replace the group with
      an empty one;
    * if the group already exists and `overwrite=False`, fail.

    From the [`zarr.open_group`
    docs](https://zarr.readthedocs.io/en/stable/api/hierarchy.html#zarr.hierarchy.open_group):

    * `mode="r"` means read only (must exist);
    * `mode="r+"` means read/write (must exist);
    * `mode="a"` means read/write (create if doesn’t exist);
    * `mode="w"` means create (overwrite if exists);
    * `mode="w-"` means create (fail if exists).


    Args:
        path:
            Store or path to directory in file system or name of zip file
            (`zarr.open_group` parameter).
        overwrite:
            Determines the `mode` parameter of `zarr.open_group`, which is
            `"w"` (if `overwrite=True`) or `"w-"` (if `overwrite=False`).
        logger:
            The logger to use (if unset, use `logging.getLogger(None)`)
        open_group_kwargs:
            Keyword arguments of `zarr.open_group`.

    Returns:
        The zarr group.

    Raises:
        OverwriteNotAllowedError:
            If `overwrite=False` and the group already exists.
    """

    # Set logger
    if logger is None:
        logger = logging.getLogger(None)

    # Set mode for zarr.open_group
    if overwrite:
        new_mode = "w"
    else:
        new_mode = "w-"

    # Raise warning if we are overriding an existing value of `mode`
    if "mode" in open_group_kwargs.keys():
        mode = open_group_kwargs.pop("mode")
        logger.warning(
            f"Overriding {mode=} with {new_mode=}, "
            "in open_zarr_group_with_overwrite"
        )

    # Call zarr.open_group
    try:
        return zarr.open_group(path, mode=new_mode, **open_group_kwargs)
    except ContainsGroupError:
        # Re-raise error with custom message and type
        error_msg = (
            f"Cannot create zarr group at {path=} with `{overwrite=}` "
            "(original error: `zarr.errors.ContainsGroupError`). "
            "Hint: try setting `overwrite=True`."
        )
        logger.error(error_msg)
        raise OverwriteNotAllowedError(error_msg)


def _write_elem_with_overwrite(
    group: zarr.Group,
    key: str,
    elem: Any,
    *,
    overwrite: bool,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Wrap `anndata.experimental.write_elem`, to include `overwrite` parameter.

    See docs for the original function
    [here](https://anndata.readthedocs.io/en/stable/generated/anndata.experimental.write_elem.html).

    This function writes `elem` to the sub-group `key` of `group`. The
    `overwrite`-related expected behavior is:

    * if the sub-group does not exist, create it (independently on
      `overwrite`);
    * if the sub-group already exists and `overwrite=True`, overwrite the
      sub-group;
    * if the sub-group already exists and `overwrite=False`, fail.

    Note that this version of the wrapper does not include the original
    `dataset_kwargs` parameter.

    Args:
        group:
            The group to write to.
        key:
            The key to write to in the group. Note that absolute paths will be
            written from the root.
        elem:
            The element to write. Typically an in-memory object, e.g. an
            AnnData, pandas dataframe, scipy sparse matrix, etc.
        overwrite:
            TBD
        logger:
            The logger to use (if unset, use `logging.getLogger(None)`)

    Raises:
        OverwriteNotAllowedError:
            If `overwrite=False` and the sub-group already exists.
    """

    # Set logger
    if logger is None:
        logger = logging.getLogger(None)

    if key in set(group.group_keys()):
        if not overwrite:
            error_msg = (
                f"Sub-group '{key}' of group {group.store.path} "
                f"already exists, but `{overwrite=}`. "
                "Hint: try setting `overwrite=True`."
            )
            logger.error(error_msg)
            raise OverwriteNotAllowedError(error_msg)
    write_elem(group, key, elem)


def write_table(
    image_group: zarr.Group,
    table_name: str,
    table: ad.AnnData,
    overwrite: bool = False,
    ngff_table_attrs: Optional[dict[str, str]] = None,
    logger: Optional[logging.Logger] = None,
) -> zarr.group:
    """
    Handle multiple options for writing an AnnData table to a zarr group.

    1. Create the `tables` group, if needed.
    2. If `overwrite=False`, check that the new table does not exist (either in
       zarr attributes or as a zarr sub-group).
    3. Call the `_write_elem_with_overwrite` wrapper with the appropriate
       `overwrite` parameter.
    4. Update the `tables` attribute of the image group.
    5. If `ngff_table_attrs` is set, include a set of attributes in the
       new-table zarr group, based on a proposed change to the OME-NGFF specs
       (https://github.com/ome/ngff/pull/64)

    Args:
        image_group:
            The group to write to.
        table_name:
            The key to write to in the group. Note that absolute paths will be
            written from the root.
        table:
            The AnnData table to write.
        overwrite:
            TBD
        ngff_table_attrs:
            If set, a dictionary with keys `region_path` (required), `type`
            (optional, defaults to `"ngff:region_table"`) and `instance_key`
            (optional, defaults to `"label"`).
        logger:
            The logger to use (if unset, use `logging.getLogger(None)`)

    Returns:
        Zarr group of the new table.
    """

    # Set logger
    if logger is None:
        logger = logging.getLogger(None)

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
                f"already exists, but `{overwrite=}`. "
                "Hint: try setting `overwrite=True`."
            )
            logger.error(error_msg)
            raise OverwriteNotAllowedError(error_msg)
        if table_name in current_tables:
            error_msg = (
                f"Item '{table_name}' already exists in `tables` attribute of "
                f"group {image_group.store.path}, but `{overwrite=}`. "
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

    # Optionally update OME-NGFF metadata for the new-table zarr group, based
    # on a proposed change to the OME-NGFF table specs
    # (https://github.com/ome/ngff/pull/64)
    if ngff_table_attrs is not None:
        # Define region attribute (with no default)
        try:
            region = dict(path=ngff_table_attrs["region_path"])
        except KeyError:
            raise ValueError(
                "If provided, `ngff_table_attrs` argument of "
                "`write_table` must include a `region_path` key. "
                f"Given: {ngff_table_attrs}."
            )
        # Define type attribute (defaulting to 'ngff:region_table')
        _type = ngff_table_attrs.get("type", "ngff:region_table")
        # Define instance_key attribute (defaulting to 'label')
        instance_key = ngff_table_attrs.get("instance_key", "label")
        # Set attributes of the new-table group
        table_group.attrs["type"] = _type
        table_group.attrs["region"] = region
        table_group.attrs["instance_key"] = instance_key

    return table_group

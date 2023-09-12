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
from zarr.errors import GroupNotFoundError


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

    # Write log about current status
    logger.info(f"Start open_zarr_group_with_overwrite ({overwrite=}).")
    try:
        # Call `zarr.open_group` with `mode="r"`, which fails for missing group
        current_group = zarr.open_group(path, mode="r")
        keys = list(current_group.group_keys())
        logger.info(f"Zarr group {path} already exists, with {keys=}")
    except GroupNotFoundError:
        logger.info(f"Zarr group {path} does not exist yet.")

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
            "(original error: `zarr.errors.ContainsGroupError`).\n"
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
            If `True`, overwrite the `key` sub-group (if present); if `False`
            and `key` sub-group exists, raise an error.
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
                f"already exists, but `{overwrite=}`.\n"
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
    table_attrs: Optional[dict[str, Any]] = None,
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
        logger:
            The logger to use (if unset, use `logging.getLogger(None)`).

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

    # Optionally update attributes of the new-table zarr group
    if table_attrs is not None:
        if table_attrs.get("type") == "ngff:region_table":
            # Verify whether we comply with a proposed change to the OME-NGFF
            # table specs (https://github.com/ome/ngff/pull/64)
            try:
                table_attrs["instance_key"]
                table_attrs["region"]["path"]
            except KeyError as e:
                logger.warning(
                    f"The table_attrs parameter of write_elem has "
                    "type='ngff:region_table' but does not comply with the "
                    "proposed table specs. "
                    f"Original error: KeyError: {str(e)}"
                )
        # Overwrite table_group attributes with table_attrs key/value pairs
        table_group.attrs.put(table_attrs)

    return table_group


def prepare_label_group(
    image_group: zarr.Group,
    label_name: str,
    overwrite: bool = False,
    label_attrs: Optional[dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> zarr.group:
    """
    Set the stage for writing labels to a zarr group

    This helper function is similar to `write_table`, in that it prepares the
    appropriate zarr groups (`labels` and the new-label one) and performs
    `overwrite`-dependent checks. At a difference with `write_table`, this
    function does not actually write the label array to the new zarr group;
    such writing operation must take place in the actual task function, since
    in fractal-tasks-core it is done sequentially on different `region`s of the
    zarr array.

    What this function does is:

    1. Create the `labels` group, if needed.
    2. If `overwrite=False`, check that the new label does not exist (either in
       zarr attributes or as a zarr sub-group).
    3. Update the `labels` attribute of the image group.
    4. If `label_attrs` is set, include this set of attributes in the
       new-label zarr group.

    Args:
        image_group:
            The group to write to.
        label_name:
            The name of the new label.
        overwrite:
            If `False`, check that the new label does not exist (either in zarr
            attributes or as a zarr sub-group); if `True` propagate parameter
            to `create_group` method, making it overwrite any existing
            sub-group with the given name.
        label_attrs:
            If set, overwrite label_group attributes with label_attrs key/value
            pairs.
        logger:
            The logger to use (if unset, use `logging.getLogger(None)`).

    Returns:
        Zarr group of the new label.
    """

    # Set logger
    if logger is None:
        logger = logging.getLogger(None)

    # Create labels group (if needed) and extract current_labels
    if "labels" not in set(image_group.group_keys()):
        labels_group = image_group.create_group("labels", overwrite=False)
    else:
        labels_group = image_group["labels"]
    current_labels = labels_group.attrs.asdict().get("labels", [])

    # If overwrite=False, check that the new label does not exist (either as a
    # zarr sub-group or as part of the zarr-group attributes)
    if not overwrite:
        if label_name in set(labels_group.group_keys()):
            error_msg = (
                f"Sub-group '{label_name}' of group {image_group.store.path} "
                f"already exists, but `{overwrite=}`.\n"
                "Hint: try setting `overwrite=True`."
            )
            logger.error(error_msg)
            raise OverwriteNotAllowedError(error_msg)
        if label_name in current_labels:
            error_msg = (
                f"Item '{label_name}' already exists in `labels` attribute of "
                f"group {image_group.store.path}, but `{overwrite=}`.\n"
                "Hint: try setting `overwrite=True`."
            )
            logger.error(error_msg)
            raise OverwriteNotAllowedError(error_msg)

    # Update the `labels` metadata of the image group, if needed
    if label_name not in current_labels:
        new_labels = current_labels + [label_name]
        labels_group.attrs["labels"] = new_labels

    # Define new-label group
    label_group = labels_group.create_group(label_name, overwrite=overwrite)

    # Optionally update attributes of the new-table zarr group
    if label_attrs is not None:
        # Overwrite label_group attributes with label_attrs key/value pairs
        label_group.attrs.put(label_attrs)

    return label_group

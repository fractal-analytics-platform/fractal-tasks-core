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

import zarr.hierarchy
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
) -> zarr.hierarchy.Group:
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
    group: zarr.hierarchy.Group,
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

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

import zarr
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

    * if `overwrite=False` and the group does not exist, create it;
    * if `overwrite=True` and the group does not exist, create it;
    * if `overwrite=True` and the group already exists, replace it with a new
      empty group;
    * if `overwrite=False` and the group already exists, fail with a relevant
      error.

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
            A `logging.Logger` instance; if unset, it will be set to
            `logging.getLogger(None)`.
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

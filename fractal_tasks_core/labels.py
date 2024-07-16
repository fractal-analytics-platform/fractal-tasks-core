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
Module which currently only hosts `prepare_label_group`.
"""
import logging
from typing import Any
from typing import Optional

import zarr.hierarchy
from pydantic import ValidationError

from fractal_tasks_core.ngff import NgffImageMeta
from fractal_tasks_core.zarr_utils import OverwriteNotAllowedError


def prepare_label_group(
    image_group: zarr.hierarchy.Group,
    label_name: str,
    label_attrs: dict[str, Any],
    overwrite: bool = False,
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
            The name of the new label; this name also overrides the multiscale
            name in NGFF-image Zarr attributes, if needed.
        overwrite:
            If `False`, check that the new label does not exist (either in zarr
            attributes or as a zarr sub-group); if `True` propagate parameter
            to `create_group` method, making it overwrite any existing
            sub-group with the given name.
        label_attrs:
            Zarr attributes of the label-image group.
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

    # Validate attrs against NGFF specs 0.4
    try:
        meta = NgffImageMeta(**label_attrs)
    except ValidationError as e:
        error_msg = (
            "Label attributes do not comply with NGFF image "
            "specifications, as encoded in fractal-tasks-core.\n"
            f"Original error:\nValidationError: {str(e)}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    # Replace multiscale name with label_name, if needed
    current_multiscale_name = meta.multiscale.name
    if current_multiscale_name != label_name:
        logger.warning(
            f"Setting multiscale name to '{label_name}' (old value: "
            f"'{current_multiscale_name}') in label-image NGFF "
            "attributes."
        )
        label_attrs["multiscales"][0]["name"] = label_name
    # Overwrite label_group attributes with label_attrs key/value pairs
    label_group.attrs.put(label_attrs)

    return label_group

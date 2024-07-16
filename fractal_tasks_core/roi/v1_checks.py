# Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Tommaso Comparin <tommaso.comparin@exact-lab.it>
# Joel Lüthi <joel.luethi@uzh.ch>
#
# This file is part of Fractal and was originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
"""
Functions to check content of ROI tables.
"""
import logging
from typing import Optional

import anndata as ad
import zarr
from pydantic import ValidationError

from fractal_tasks_core.tables.v1 import MaskingROITableAttrs


logger = logging.getLogger(__name__)


def check_valid_ROI_indices(
    list_indices: list[list[int]],
    ROI_table_name: str,
) -> None:
    """
    Check that list of indices has zero origin on each axis.

    See fractal-tasks-core issues #530 and #554.

    This helper function is meant to provide informative error messages when
    ROI tables created with fractal-tasks-core up to v0.11 are used in v0.12.
    This function will be deprecated and removed as soon as the v0.11/v0.12
    transition advances.

    Note that only `FOV_ROI_table` and `well_ROI_table` have to fulfill this
    constraint, while ROI tables obtained through segmentation may have
    arbitrary (non-negative) indices.

    Args:
        list_indices:
            Output of `convert_ROI_table_to_indices`; each item is like
            `[start_z, end_z, start_y, end_y, start_x, end_x]`.
        ROI_table_name: Name of the ROI table.

    Raises:
        ValueError:
            If the table name is `FOV_ROI_table` or `well_ROI_table` and the
                minimum value of `start_x`, `start_y` and `start_z` are not all
                zero.
    """
    if ROI_table_name not in ["FOV_ROI_table", "well_ROI_table"]:
        # This validation function only applies to the FOV/well ROI tables
        # generated with fractal-tasks-core
        return

    # Find minimum index along ZYX
    min_start_z = min(item[0] for item in list_indices)
    min_start_y = min(item[2] for item in list_indices)
    min_start_x = min(item[4] for item in list_indices)

    # Check that minimum indices are all zero
    for ind, min_index in enumerate((min_start_z, min_start_y, min_start_x)):
        if min_index != 0:
            axis = ["Z", "Y", "X"][ind]
            raise ValueError(
                f"{axis} component of ROI indices for table `{ROI_table_name}`"
                f" do not start with 0, but with {min_index}.\n"
                "Hint: As of fractal-tasks-core v0.12, FOV/well ROI "
                "tables with non-zero origins (e.g. the ones created with "
                "v0.11) are not supported."
            )


def is_ROI_table_valid(*, table_path: str, use_masks: bool) -> Optional[bool]:
    """
    Verify some validity assumptions on a ROI table.

    This function reflects our current working assumptions (e.g. the presence
    of some specific columns); this may change in future versions.

    If `use_masks=True`, we verify that the table is a valid
    `masking_roi_table` as of table specifications V1; if this check fails,
    `use_masks` should be set to `False` upstream in the parent function.

    Args:
        table_path: Path of the AnnData ROI table to be checked.
        use_masks: If `True`, perform some additional checks related to
            masked loading.

    Returns:
        Always `None` if `use_masks=False`, otherwise return whether the table
            is valid for masked loading.
    """

    table = ad.read_zarr(table_path)
    are_ROI_table_columns_valid(table=table)
    if not use_masks:
        return None

    # Check whether the table can be used for masked loading
    attrs = zarr.group(table_path).attrs.asdict()
    logger.info(f"ROI table at {table_path} has attrs: {attrs}")
    try:
        MaskingROITableAttrs(**attrs)
        logging.info("ROI table can be used for masked loading")
        return True
    except ValidationError:
        logging.info("ROI table cannot be used for masked loading")
        return False


def are_ROI_table_columns_valid(*, table: ad.AnnData) -> None:
    """
    Verify some validity assumptions on a ROI table.

    This function reflects our current working assumptions (e.g. the presence
    of some specific columns); this may change in future versions.

    Args:
        table: AnnData table to be checked
    """

    # Hard constraint: table columns must include some expected ones
    columns = [
        "x_micrometer",
        "y_micrometer",
        "z_micrometer",
        "len_x_micrometer",
        "len_y_micrometer",
        "len_z_micrometer",
    ]
    for column in columns:
        if column not in table.var_names:
            raise ValueError(f"Column {column} is not present in ROI table")

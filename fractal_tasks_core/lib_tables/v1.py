"""
Functions and classes related to table specifications V1 (see
https://fractal-analytics-platform.github.io/fractal-tasks-core/tables).
"""
import logging
import warnings
from typing import Any
from typing import Literal
from typing import Optional

import anndata as ad
import zarr.hierarchy
from pydantic import BaseModel
from pydantic import validator
from pydantic.error_wrappers import ValidationError

from ..lib_write import _write_elem_with_overwrite
from ..lib_write import OverwriteNotAllowedError

logger = logging.getLogger(__name__)


class _RegionType(BaseModel):
    path: str


class MaskingROITableAttrs(BaseModel):
    type: Literal["masking_roi_table", "ngff:region_table"]
    region: _RegionType
    instance_key: str

    @validator("type", always=True)
    def warning_for_old_table_type(cls, v):
        if v == "ngff:region_table":
            warning_msg = (
                "Table type `ngff:region_table` is currently accepted instead "
                "of `masking_roi_table`, but it will be deprecated in the "
                "future. Please switch to `type='masking_roi_table'`."
            )

            warnings.warn(warning_msg, FutureWarning)
        return v


class FeatureTableAttrs(BaseModel):
    type: Literal["feature_table", "ngff:region_table"]
    region: _RegionType
    instance_key: str

    @validator("type", always=True)
    def warning_for_old_table_type(cls, v):
        if v == "ngff:region_table":
            warning_msg = (
                "Table type `ngff:region_table` is currently accepted instead "
                "of `feature_table`, but it will be deprecated in the "
                "future. Please switch to `type='masking_roi_table'`."
            )

            warnings.warn(warning_msg, FutureWarning)
        return v


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
        table_attrs = dict(fractal_table_version="1")
    elif table_attrs.get("fractal_table_version", None) is None:
        table_attrs["fractal_table_version"] = "1"

    # Raise warning for non-compliance with table specs
    table_type = table_attrs.get("type", None)
    if table_type is None:
        pass
    elif table_type == "roi_table":
        pass
    elif table_type == "masking_roi_table":
        try:
            MaskingROITableAttrs(**table_attrs)
        except ValidationError as e:
            logger.warning(
                f"Current `masking_roi_table` does not comply with Fractal "
                f"table specs V1. Original error: ValidationError: {str(e)}"
            )
    elif table_type == "feature_table":
        try:
            FeatureTableAttrs(**table_attrs)
        except ValidationError as e:
            logger.warning(
                f"Current `feature_table` does not comply with Fractal "
                f"table specs V1. Original error: ValidationError: {str(e)}"
            )
    else:
        logger.warning(f"Unknown table type `{table_type}`.")

    # Update table_group attributes with table_attrs key/value pairs
    table_group.attrs.update(**table_attrs)

    return table_group

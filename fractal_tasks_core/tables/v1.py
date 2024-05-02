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
from anndata.experimental import write_elem
from pydantic import BaseModel
from pydantic import field_validator
from pydantic import ValidationError

from fractal_tasks_core.zarr_utils import OverwriteNotAllowedError

logger = logging.getLogger(__name__)


class _RegionType(BaseModel):
    path: str


class MaskingROITableAttrs(BaseModel):
    type: Literal["masking_roi_table", "ngff:region_table"]
    region: _RegionType
    instance_key: str

    @field_validator("type")
    @classmethod
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

    @field_validator("type")
    @classmethod
    def warning_for_old_table_type(cls, v):
        if v == "ngff:region_table":
            warning_msg = (
                "Table type `ngff:region_table` is currently accepted instead "
                "of `feature_table`, but it will be deprecated in the "
                "future. Please switch to `type='feature_table'`."
            )

            warnings.warn(warning_msg, FutureWarning)
        return v


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


def _write_table_v1(
    image_group: zarr.hierarchy.Group,
    table_name: str,
    table: ad.AnnData,
    overwrite: bool = False,
    table_type: Optional[str] = None,
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
    5. Validate `table_type` and `table_attrs` according to Fractal table
       specifications, and raise errors/warnings if needed; then set the
       appropriate attributes in the new-table Zarr group.


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
        table_type: `type` attribute for the table; in case `type` is also
            present in `table_attrs`, this function argument takes priority.
        table_attrs:
            If set, overwrite table_group attributes with table_attrs key/value
            pairs. If `table_type` is not provided, then `table_attrs` must
            include the `type` key.

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

    # Always include fractal-roi-table version in table attributes
    if table_attrs is None:
        table_attrs = dict(fractal_table_version="1")
    elif table_attrs.get("fractal_table_version", None) is None:
        table_attrs["fractal_table_version"] = "1"

    # Set type attribute for the table
    table_type_from_attrs = table_attrs.get("type", None)
    if table_type is not None:
        if table_type_from_attrs is not None:
            logger.warning(
                f"Setting table type to '{table_type}' (and overriding "
                f"'{table_type_from_attrs}' attribute)."
            )
        table_attrs["type"] = table_type
    else:
        if table_type_from_attrs is None:
            raise ValueError(
                "Missing attribute `type` for table; this must be provided"
                " either via `table_type` or within `table_attrs`."
            )

    # Prepare/validate attributes for the table
    table_type = table_attrs.get("type", None)
    if table_type == "roi_table":
        pass
    elif table_type == "masking_roi_table":
        try:
            MaskingROITableAttrs(**table_attrs)
        except ValidationError as e:
            error_msg = (
                "Table attributes do not comply with Fractal "
                "`masking_roi_table` specifications V1.\nOriginal error:\n"
                f"ValidationError: {str(e)}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
    elif table_type == "feature_table":
        try:
            FeatureTableAttrs(**table_attrs)
        except ValidationError as e:
            error_msg = (
                "Table attributes do not comply with Fractal "
                "`feature_table` specifications V1.\nOriginal error:\n"
                f"ValidationError: {str(e)}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
    else:
        logger.warning(f"Unknown table type `{table_type}`.")

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

    # Update table_group attributes with table_attrs key/value pairs
    table_group.attrs.update(**table_attrs)

    return table_group


def get_tables_list_v1(
    zarr_url: str, table_type: str = None, strict: bool = False
) -> list[str]:
    """
    Find the list of tables in the Zarr file

    Optionally match a table type and only return the names of those tables.

    Args:
        zarr_url: Path to the OME-Zarr image
        table_type: The type of table to look for. Special handling for
            "ROIs" => matches both "roi_table" & "masking_roi_table".
        strict: If `True`, only return tables that have a type attribute.
            If `False`, also include tables without a type attribute.

    Returns:
        List of the names of available tables
    """
    with zarr.open(zarr_url, mode="r") as zarr_group:
        zarr_subgroups = list(zarr_group.group_keys())
    if "tables" not in zarr_subgroups:
        return []
    with zarr.open(zarr_url, mode="r") as zarr_group:
        all_tables = list(zarr_group.tables.group_keys())

    if not table_type:
        return all_tables
    else:
        return _filter_tables_by_type_v1(
            zarr_url, all_tables, table_type, strict
        )


def _filter_tables_by_type_v1(
    zarr_url: str,
    all_tables: list[str],
    table_type: Optional[str] = None,
    strict: bool = False,
) -> list[str]:
    tables_list = []
    for table_name in all_tables:
        with zarr.open(zarr_url, mode="r").tables[table_name] as table:
            table_attrs = table.attrs.asdict()
            if "type" in table_attrs:
                if table_type == "ROIs":
                    roi_table_types = ["roi_table", "masking_roi_table"]
                    if table_attrs["type"] in roi_table_types:
                        tables_list.append(table_name)
                elif table_attrs["type"] == table_type:
                    tables_list.append(table_name)
            else:
                # If there are tables without types, let the users choose
                # from all tables
                logger.warning(f"Table {table_name} had no type attribute.")
                if not strict:
                    tables_list.append(table_name)
    return tables_list

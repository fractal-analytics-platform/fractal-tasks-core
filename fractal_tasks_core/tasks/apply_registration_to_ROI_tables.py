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
Applies the multiplexing translation to all ROI tables
"""
import copy
import logging
from typing import Any
from typing import Optional
from typing import Sequence

import anndata as ad
import numpy as np
import pandas as pd
import zarr
from pydantic.decorator import validate_arguments

from fractal_tasks_core.ngff import load_NgffWellMeta
from fractal_tasks_core.roi import (
    are_ROI_table_columns_valid,
)
from fractal_tasks_core.tables import write_table

logger = logging.getLogger(__name__)


@validate_arguments
def apply_registration_to_ROI_tables(
    *,
    # Fractal arguments
    input_paths: Sequence[str],
    output_path: str,
    component: str,
    metadata: dict[str, Any],
    # Task-specific arguments
    roi_table: str = "FOV_ROI_table",
    reference_cycle: int = 0,
    new_roi_table: Optional[str] = None,
) -> dict[str, Any]:
    """
    Applies pre-calculated registration to ROI tables.

    Apply pre-calculated registration such that resulting ROIs contain
    the consensus align region between all cycles.

    Parallelization level: well

    Args:
        input_paths: List of input paths where the image data is stored as
            OME-Zarrs. Should point to the parent folder containing one or many
            OME-Zarr files, not the actual OME-Zarr file. Example:
            `["/some/path/"]`. This task only supports a single input path.
            (standard argument for Fractal tasks, managed by Fractal server).
        output_path: This parameter is not used by this task.
            (standard argument for Fractal tasks, managed by Fractal server).
        component: Path to the OME-Zarr image in the OME-Zarr plate that is
            processed. Example: `"some_plate.zarr/B/03/0"`.
            (standard argument for Fractal tasks, managed by Fractal server).
        metadata: This parameter is not used by this task.
            (standard argument for Fractal tasks, managed by Fractal server).
        roi_table: Name of the ROI table over which the task loops to
            calculate the registration. Examples: `FOV_ROI_table` => loop over
            the field of views, `well_ROI_table` => process the whole well as
            one image.
        reference_cycle: Which cycle to register against. Defaults to 0,
            which is the first OME-Zarr image in the well, usually the first
            cycle that was provided
        new_roi_table: Optional name for the new, registered ROI table. If no
            name is given, it will default to "registered_" + `roi_table`

    """
    if not new_roi_table:
        new_roi_table = "registered_" + roi_table
    logger.info(
        f"Running for {input_paths=}, {component=}. \n"
        f"Applyg translation registration to {roi_table=} and storing it as "
        f"{new_roi_table=}."
    )

    well_zarr = f"{input_paths[0]}/{component}"
    ngff_well_meta = load_NgffWellMeta(well_zarr)
    acquisition_dict = ngff_well_meta.get_acquisition_paths()
    logger.info(
        "Calculating common registration for the following cycles: "
        f"{acquisition_dict}"
    )

    # TODO: Allow a filter on which acquisitions should get processed?

    # Collect all the ROI tables
    roi_tables = {}
    roi_tables_attrs = {}
    for acq in acquisition_dict.keys():
        acq_path = acquisition_dict[acq]
        curr_ROI_table = ad.read_zarr(
            f"{well_zarr}/{acq_path}/tables/{roi_table}"
        )
        curr_ROI_table_group = zarr.open_group(
            f"{well_zarr}/{acq_path}/tables/{roi_table}", mode="r"
        )
        curr_ROI_table_attrs = curr_ROI_table_group.attrs.asdict()

        # For reference_cycle acquisition, handle the fact that it doesn't
        # have the shifts
        if acq == reference_cycle:
            curr_ROI_table = add_zero_translation_columns(curr_ROI_table)
        # Check for valid ROI tables
        are_ROI_table_columns_valid(table=curr_ROI_table)
        translation_columns = [
            "translation_z",
            "translation_y",
            "translation_x",
        ]
        if curr_ROI_table.var.index.isin(translation_columns).sum() != 3:
            raise ValueError(
                f"Cycle {acq}'s {roi_table} does not contain the "
                f"translation columns {translation_columns} necessary to use "
                "this task."
            )
        roi_tables[acq] = curr_ROI_table
        roi_tables_attrs[acq] = curr_ROI_table_attrs

    # Check that all acquisitions have the same ROIs
    rois = roi_tables[reference_cycle].obs.index
    for acq, acq_roi_table in roi_tables.items():
        if not (acq_roi_table.obs.index == rois).all():
            raise ValueError(
                f"Acquisition {acq} does not contain the same ROIs as the "
                f"reference acquisition {reference_cycle}:\n"
                f"{acq}: {acq_roi_table.obs.index}\n"
                f"{reference_cycle}: {rois}"
            )

    roi_table_dfs = [
        roi_table.to_df().loc[:, translation_columns]
        for roi_table in roi_tables.values()
    ]
    logger.info("Calculating min & max translation across cycles.")
    max_df, min_df = calculate_min_max_across_dfs(roi_table_dfs)
    shifted_rois = {}
    # Loop over acquisitions
    for acq in acquisition_dict.keys():
        shifted_rois[acq] = apply_registration_to_single_ROI_table(
            roi_tables[acq], max_df, min_df
        )

        # TODO: Drop translation columns from this table?

        logger.info(
            f"Write the registered ROI table {new_roi_table} for {acq=}"
        )
        # Save the shifted ROI table as a new table
        image_group = zarr.group(f"{well_zarr}/{acq}")
        write_table(
            image_group,
            new_roi_table,
            shifted_rois[acq],
            table_attrs=roi_tables_attrs[acq],
        )

    # TODO: Optionally apply registration to other tables as well?
    # e.g. to well_ROI_table based on FOV_ROI_table
    # => out of scope for the initial task, apply registration separately
    # to each table
    # Easiest implementation: Apply average shift calculcated here to other
    # ROIs. From many to 1 (e.g. FOV => well) => average shift, but crop len
    # From well to many (e.g. well to FOVs) => average shift, crop len by that
    # amount
    # Many to many (FOVs to organoids) => tricky because of matching

    return {}


# Helper functions
def add_zero_translation_columns(ad_table: ad.AnnData):
    """
    Add three zero-filled columns (`translation_{x,y,z}`) to an AnnData table.
    """
    columns = ["translation_z", "translation_y", "translation_x"]
    if ad_table.var.index.isin(columns).any().any():
        raise ValueError(
            "The roi table already contains translation columns. Did you "
            "enter a wrong reference cycle?"
        )
    df = pd.DataFrame(np.zeros([len(ad_table), 3]), columns=columns)
    df.index = ad_table.obs.index
    ad_new = ad.concat([ad_table, ad.AnnData(df)], axis=1)
    return ad_new


def calculate_min_max_across_dfs(tables_list):
    # Initialize dataframes to store the max and min values
    max_df = pd.DataFrame(
        index=tables_list[0].index, columns=tables_list[0].columns
    )
    min_df = pd.DataFrame(
        index=tables_list[0].index, columns=tables_list[0].columns
    )

    # Loop through the tables and calculate max and min values
    for table in tables_list:
        max_df = pd.DataFrame(
            np.maximum(max_df.values, table.values),
            columns=max_df.columns,
            index=max_df.index,
        )
        min_df = pd.DataFrame(
            np.minimum(min_df.values, table.values),
            columns=min_df.columns,
            index=min_df.index,
        )

    return max_df, min_df


def apply_registration_to_single_ROI_table(
    roi_table: ad.AnnData,
    max_df: pd.DataFrame,
    min_df: pd.DataFrame,
) -> ad.AnnData:
    """
    Applies the registration to a ROI table

    Calculates the new position as: p = position + max(shift, 0) - own_shift
    Calculates the new len as: l = len - max(shift, 0) + min(shift, 0)

    Args:
        roi_table: AnnData table which contains a Fractal ROI table.
            Rows are ROIs
        max_df: Max translation shift in z, y, x for each ROI. Rows are ROIs,
            columns are translation_z, translation_y, translation_x
        min_df: Min translation shift in z, y, x for each ROI. Rows are ROIs,
            columns are translation_z, translation_y, translation_x
    Returns:
        ROI table where all ROIs are registered to the smallest common area
        across all cycles.
    """
    roi_table = copy.deepcopy(roi_table)
    rois = roi_table.obs.index
    if (rois != max_df.index).all() or (rois != min_df.index).all():
        raise ValueError(
            "ROI table and max & min translation need to contain the same "
            f"ROIS, but they were {rois=}, {max_df.index=}, {min_df.index=}"
        )

    for roi in rois:
        roi_table[[roi], ["z_micrometer"]] = (
            roi_table[[roi], ["z_micrometer"]].X
            + float(max_df.loc[roi, "translation_z"])
            - roi_table[[roi], ["translation_z"]].X
        )
        roi_table[[roi], ["y_micrometer"]] = (
            roi_table[[roi], ["y_micrometer"]].X
            + float(max_df.loc[roi, "translation_y"])
            - roi_table[[roi], ["translation_y"]].X
        )
        roi_table[[roi], ["x_micrometer"]] = (
            roi_table[[roi], ["x_micrometer"]].X
            + float(max_df.loc[roi, "translation_x"])
            - roi_table[[roi], ["translation_x"]].X
        )
        # This calculation only works if all ROIs are the same size initially!
        roi_table[[roi], ["len_z_micrometer"]] = (
            roi_table[[roi], ["len_z_micrometer"]].X
            - float(max_df.loc[roi, "translation_z"])
            + float(min_df.loc[roi, "translation_z"])
        )
        roi_table[[roi], ["len_y_micrometer"]] = (
            roi_table[[roi], ["len_y_micrometer"]].X
            - float(max_df.loc[roi, "translation_y"])
            + float(min_df.loc[roi, "translation_y"])
        )
        roi_table[[roi], ["len_x_micrometer"]] = (
            roi_table[[roi], ["len_x_micrometer"]].X
            - float(max_df.loc[roi, "translation_x"])
            + float(min_df.loc[roi, "translation_x"])
        )
    return roi_table


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=apply_registration_to_ROI_tables,
        logger_name=logger.name,
    )

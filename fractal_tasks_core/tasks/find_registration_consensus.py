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
import logging
from typing import Optional

import anndata as ad
import zarr
from pydantic import validate_call

from fractal_tasks_core.roi import (
    are_ROI_table_columns_valid,
)
from fractal_tasks_core.tables import write_table
from fractal_tasks_core.tasks._registration_utils import (
    add_zero_translation_columns,
)
from fractal_tasks_core.tasks._registration_utils import (
    apply_registration_to_single_ROI_table,
)
from fractal_tasks_core.tasks._registration_utils import (
    calculate_min_max_across_dfs,
)
from fractal_tasks_core.tasks.io_models import InitArgsRegistrationConsensus


logger = logging.getLogger(__name__)


@validate_call
def find_registration_consensus(
    *,
    # Fractal parameters
    zarr_url: str,
    init_args: InitArgsRegistrationConsensus,
    # Core parameters
    roi_table: str = "FOV_ROI_table",
    # Advanced parameters
    new_roi_table: Optional[str] = None,
):
    """
    Applies pre-calculated registration to ROI tables.

    Apply pre-calculated registration such that resulting ROIs contain
    the consensus align region between all acquisitions.

    Parallelization level: well

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            Refers to the zarr_url of the reference acquisition.
            (standard argument for Fractal tasks, managed by Fractal server).
        init_args: Intialization arguments provided by
            `init_group_by_well_for_multiplexing`. It contains the
            zarr_url_list listing all the zarr_urls in the same well as the
            zarr_url of the reference acquisition that are being processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        roi_table: Name of the ROI table over which the task loops to
            calculate the registration. Examples: `FOV_ROI_table` => loop over
            the field of views, `well_ROI_table` => process the whole well as
            one image.
        new_roi_table: Optional name for the new, registered ROI table. If no
            name is given, it will default to "registered_" + `roi_table`

    """
    if not new_roi_table:
        new_roi_table = "registered_" + roi_table
    logger.info(
        f"Running for {zarr_url=} & the other acquisitions in that well. \n"
        f"Applying translation registration to {roi_table=} and storing it as "
        f"{new_roi_table=}."
    )

    # Collect all the ROI tables
    roi_tables = {}
    roi_tables_attrs = {}
    for acq_zarr_url in init_args.zarr_url_list:
        curr_ROI_table = ad.read_zarr(f"{acq_zarr_url}/tables/{roi_table}")
        curr_ROI_table_group = zarr.open_group(
            f"{acq_zarr_url}/tables/{roi_table}", mode="r"
        )
        curr_ROI_table_attrs = curr_ROI_table_group.attrs.asdict()

        # For reference_acquisition, handle the fact that it doesn't
        # have the shifts
        if acq_zarr_url == zarr_url:
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
                f"{roi_table=} in {acq_zarr_url} does not contain the "
                f"translation columns {translation_columns} necessary to use "
                "this task."
            )
        roi_tables[acq_zarr_url] = curr_ROI_table
        roi_tables_attrs[acq_zarr_url] = curr_ROI_table_attrs

    # Check that all acquisitions have the same ROIs
    rois = roi_tables[list(roi_tables.keys())[0]].obs.index
    for acq_zarr_url, acq_roi_table in roi_tables.items():
        if not (acq_roi_table.obs.index == rois).all():
            raise ValueError(
                f"Acquisition {acq_zarr_url} does not contain the same ROIs "
                f"as the reference acquisition {zarr_url}:\n"
                f"{acq_zarr_url}: {acq_roi_table.obs.index}\n"
                f"{zarr_url}: {rois}"
            )

    roi_table_dfs = [
        roi_table.to_df().loc[:, translation_columns]
        for roi_table in roi_tables.values()
    ]
    logger.info("Calculating min & max translation across acquisitions.")
    max_df, min_df = calculate_min_max_across_dfs(roi_table_dfs)
    shifted_rois = {}

    # Loop over acquisitions
    for acq_zarr_url in init_args.zarr_url_list:
        shifted_rois[acq_zarr_url] = apply_registration_to_single_ROI_table(
            roi_tables[acq_zarr_url], max_df, min_df
        )

        # TODO: Drop translation columns from this table?

        logger.info(
            f"Write the registered ROI table {new_roi_table} for "
            "{acq_zarr_url=}"
        )
        # Save the shifted ROI table as a new table
        image_group = zarr.group(acq_zarr_url)
        write_table(
            image_group,
            new_roi_table,
            shifted_rois[acq_zarr_url],
            table_attrs=roi_tables_attrs[acq_zarr_url],
            overwrite=True,
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


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=find_registration_consensus,
        logger_name=logger.name,
    )

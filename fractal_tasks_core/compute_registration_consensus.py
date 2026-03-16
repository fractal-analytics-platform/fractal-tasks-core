# Copyright 2022-2026 (C) BioVisionCenter, University of Zurich
"""
Applies the multiplexing translation to all ROI tables
"""

import logging
from typing import Literal

from ngio import Roi, open_ome_zarr_container
from ngio.tables import GenericRoiTable, RoiTable
from pydantic import validate_call

from fractal_tasks_core._registration_utils import InitArgsRegistrationConsensus
from fractal_tasks_core._utils import format_template_name

logger = logging.getLogger("compute_registration_consensus")


def _validate_if_translation_exists(tables: list[GenericRoiTable]) -> None:
    """Check that at least one table contains pre-calculated translation fields.

    The reference acquisition's table intentionally has no translations (Task 1
    only runs on non-reference acquisitions), so we require at least one table
    — not all tables — to carry the fields.  If none do, Task 1 has not been
    run yet.
    """
    tables_with_translations = 0
    for table in tables:
        for roi in table.rois():
            extra = roi.model_extra or {}
            if (
                "translation_z" in extra
                and "translation_y" in extra
                and "translation_x" in extra
            ):
                tables_with_translations += 1
                break

    if tables_with_translations == len(tables) - 1:
        # Normal case: all non-reference tables contain translations
        # and the reference table does not.
        return None

    if tables_with_translations == 0:
        raise ValueError(
            "No registration translations found in any acquisition ROI table. "
            'Please run "Calculate Registration (image-based)" before '
            '"Find Registration Consensus".'
        )
    # Edge case: some but not all non-reference tables contain translations
    if tables_with_translations != len(tables) - 1:
        raise ValueError(
            "Some but not all non-reference acquisitions contain registration "
            "translations. Something went wrong, please re-run "
            '"Calculate Registration (image-based)" and make sure it completes '
            "without errors. "
        )


def _get_roi_translation(roi: Roi, dim: Literal["z", "y", "x"]) -> float:
    """Return the translation for `dim` ("z", "y", or "x") from a ROI's extra
    fields, defaulting to 0.0 when the field is absent (e.g. reference
    acquisition whose table has no pre-computed shifts).
    """
    return (roi.model_extra or {}).get(f"translation_{dim}", 0.0)


def _group_roi_by_name(tables: list[GenericRoiTable]) -> dict[str, list[Roi]]:
    roi_dict: dict[str, list[Roi]] = {}
    for table in tables:
        for roi in table.rois():
            if roi.name is None:
                raise ValueError(f"ROI without a name found in table: {roi}")
            if roi.name not in roi_dict:
                roi_dict[roi.name] = []
            roi_dict[roi.name].append(roi)
    return roi_dict


def _find_roi_consensus(rois: list[Roi]) -> Roi:
    """
    Given a list of ROIs across acquisitions, find the consensus ROI that is
    contained in all acquisitions after applying the pre-calculated shifts.

    The consensus ROI is calculated by finding the max translation in positive
    direction and the min translation in negative direction across acquisitions
    for each roi, and applying this shift to the original ROIs to find the
    consensus region.

    Args:
        rois: List of ROIs across acquisitions with the same name that need to be
            registered to each other. They need to contain the pre-calculated
            shifts as "translation_z", "translation_y", "translation_x" in
            their metadata.
    Returns:
        Consensus ROI whose position is `base_pos + max_translation` and whose
        size is reduced by `max_translation - min_translation` in each axis.
        The position encodes the max-shift offset so that _shift_roi can derive
        each acquisition's aligned position as `consensus.pos - own_translation`.
    """
    translations_z = [_get_roi_translation(r, "z") for r in rois]
    translations_y = [_get_roi_translation(r, "y") for r in rois]
    translations_x = [_get_roi_translation(r, "x") for r in rois]

    max_z, min_z = max(translations_z), min(translations_z)
    max_y, min_y = max(translations_y), min(translations_y)
    max_x, min_x = max(translations_x), min(translations_x)

    # All ROIs in the list share the same base position (same physical FOV
    # across acquisitions), so rois[0] is a valid base for geometry.
    base = rois[0]

    z_slice = base["z"]
    z_start = (z_slice.start if z_slice.start is not None else 0.0) + max_z
    z_length = (z_slice.length if z_slice.length is not None else 0.0) - max_z + min_z
    base = base.update_slice(name="z", new_slice=(z_start, z_length))

    y_slice = base["y"]
    y_start = (y_slice.start if y_slice.start is not None else 0.0) + max_y
    y_length = (y_slice.length if y_slice.length is not None else 0.0) - max_y + min_y
    base = base.update_slice(name="y", new_slice=(y_start, y_length))
    x_slice = base["x"]
    x_start = (x_slice.start if x_slice.start is not None else 0.0) + max_x
    x_length = (x_slice.length if x_slice.length is not None else 0.0) - max_x + min_x
    base = base.update_slice(name="x", new_slice=(x_start, x_length))
    return base


def _apply_consensus_to_roi(roi: Roi, consensus_roi: Roi) -> Roi:
    """
    Given a Roi and the consensus_roi across acquisitions, calculate the
    shifted and cropped Roi that needs to be applied to the original Roi to get
    the consensus region.

    The position is computed as: consensus_roi.pos - own_translation
    which equals: base_pos + max_translation - own_translation
    The size is taken directly from the consensus_roi (same for all acquisitions).
    """
    own_z = _get_roi_translation(roi, "z")
    own_y = _get_roi_translation(roi, "y")
    own_x = _get_roi_translation(roi, "x")

    z_slice = consensus_roi["z"]
    z_start = (z_slice.start if z_slice.start is not None else 0.0) - own_z
    z_length = z_slice.length
    roi = roi.update_slice(name="z", new_slice=(z_start, z_length))

    y_slice = consensus_roi["y"]
    y_start = (y_slice.start if y_slice.start is not None else 0.0) - own_y
    y_length = y_slice.length
    roi = roi.update_slice(name="y", new_slice=(y_start, y_length))
    x_slice = consensus_roi["x"]
    x_start = (x_slice.start if x_slice.start is not None else 0.0) - own_x
    x_length = x_slice.length
    roi = roi.update_slice(name="x", new_slice=(x_start, x_length))
    return roi


def _apply_consensus_to_roi_table(
    roi_table: GenericRoiTable, consensus_rois: dict[str, Roi]
) -> RoiTable:
    """
    Given a roi_table and the consensus_rois across acquisitions, calculate the
    shifted and cropped roi_table that needs to be applied to the original
    roi_table to get the consensus region.
    """
    shifted_rois = []
    for roi in roi_table.rois():
        if roi.name is None:
            raise ValueError(f"ROI without a name found in table: {roi}")
        consensus_roi = consensus_rois[roi.name]
        shifted_rois.append(_apply_consensus_to_roi(roi, consensus_roi))
    return RoiTable(rois=shifted_rois)


@validate_call
def compute_registration_consensus(
    *,
    # Fractal parameters
    zarr_url: str,
    init_args: InitArgsRegistrationConsensus,
    # Core parameters
    input_roi_table: str = "FOV_ROI_table",
    # Advanced parameters
    registered_roi_table: str = "{input_roi_table}_registered",
) -> None:
    """
    Applies pre-calculated registration to ROI tables.

    Adjusts the ROI tables for each acquisition so that the resulting ROIs
    cover only the region visible in all acquisitions after registration.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            Refers to the zarr_url of the reference acquisition.
            (standard argument for Fractal tasks, managed by Fractal server).
        init_args: Initialization arguments provided by
            `init_registration_consensus`. It contains the
            zarr_url_list listing all the zarr_urls in the same well as the
            zarr_url of the reference acquisition that are being processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        input_roi_table: Name of the ROI table used as input for the
            "Calculate Registration (image-based)" task, which contains the
            pre-calculated translations.
        registered_roi_table: Name template for the registered ROI table. May
            contain the placeholder "{input_roi_table}", which is replaced by the
            value of "input_roi_table".

    """
    registered_roi_table = format_template_name(
        registered_roi_table, input_roi_table=input_roi_table
    )
    if registered_roi_table == input_roi_table:
        raise ValueError(
            f"registered_roi_table ({registered_roi_table!r}) must differ from "
            f"input_roi_table ({input_roi_table!r}). Overwriting the input table "
            "would destroy the pre-calculated translation shifts."
        )
    logger.info(
        f"Running for {zarr_url=} & the other acquisitions in that well. \n"
        f"Applying translation registration to {input_roi_table=} and storing it as "
        f"{registered_roi_table=}."
    )
    # Open all containers and load ROI tables, keeping containers for reuse
    # during the write phase. Reference acquisition is handled first so that
    # _find_roi_consensus receives its ROIs (zero-shift) as rois[0].
    ref_ome_zarr = open_ome_zarr_container(zarr_url)
    containers = {zarr_url: ref_ome_zarr}
    tables = {zarr_url: ref_ome_zarr.get_generic_roi_table(input_roi_table)}
    for acq_zarr_url in init_args.zarr_url_list:
        if acq_zarr_url == zarr_url:
            continue
        acq_ome_zarr = open_ome_zarr_container(acq_zarr_url)
        containers[acq_zarr_url] = acq_ome_zarr
        tables[acq_zarr_url] = acq_ome_zarr.get_generic_roi_table(input_roi_table)

    # Check if all tables contain the pre-calculated translation fields, which
    # are required for the consensus calculation.
    _validate_if_translation_exists(list(tables.values()))

    # Validate that all acquisitions have the same set of ROI names
    ref_roi_names = {roi.name for roi in list(tables.values())[0].rois()}
    for acq_zarr_url, table in tables.items():
        acq_roi_names = {roi.name for roi in table.rois()}
        if acq_roi_names != ref_roi_names:
            raise ValueError(
                f"Acquisition {acq_zarr_url} does not contain the same ROIs "
                f"as the reference acquisition {zarr_url}:\n"
                f"{acq_zarr_url}: {acq_roi_names}\n"
                f"{zarr_url}: {ref_roi_names}"
            )

    rois_by_name = _group_roi_by_name(list(tables.values()))
    consensus_rois = {
        name: _find_roi_consensus(rois) for name, rois in rois_by_name.items()
    }

    for acq_zarr_url, table in tables.items():
        registered_table = _apply_consensus_to_roi_table(table, consensus_rois)
        # If backend is none the table will default to "anndata",
        # for backwards compatibility with old tables that don't have the backend field.
        backend = table.backend_name or "anndata"
        containers[acq_zarr_url].add_table(
            name=registered_roi_table,
            table=registered_table,
            overwrite=True,
            backend=backend,
        )


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=compute_registration_consensus,
        logger_name=logger.name,
    )

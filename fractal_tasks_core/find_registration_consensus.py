# Copyright 2022-2026 (C) BioVisionCenter, University of Zurich
"""
Applies the multiplexing translation to all ROI tables
"""

import logging
from typing import Literal

from ngio import Roi, open_ome_zarr_container
from ngio.tables import GenericRoiTable, RoiTable
from pydantic import validate_call

from fractal_tasks_core._io_models import InitArgsRegistrationConsensus

logger = logging.getLogger("find_registration_consensus")


def _get_roi_translation(roi: Roi, dim: Literal["z", "y", "x"]) -> float:
    """Return the translation for `dim` ('z', 'y', or 'x') from a ROI's extra
    fields, defaulting to 0.0 when the field is absent (e.g. reference
    acquisition whose table has no pre-computed shifts).
    """
    return (roi.model_extra or {}).get(f"translation_{dim}", 0.0) or 0.0


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
    return base.model_copy(
        update={
            "z": (base.z if base.z is not None else 0.0) + max_z,
            "y": base.y + max_y,
            "x": base.x + max_x,
            "z_length": (base.z_length if base.z_length is not None else 0.0)
            - max_z
            + min_z,
            "y_length": base.y_length - max_y + min_y,
            "x_length": base.x_length - max_x + min_x,
        }
    )


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

    return roi.model_copy(
        update={
            "z": (consensus_roi.z if consensus_roi.z is not None else 0.0) - own_z,
            "y": consensus_roi.y - own_y,
            "x": consensus_roi.x - own_x,
            "z_length": consensus_roi.z_length,
            "y_length": consensus_roi.y_length,
            "x_length": consensus_roi.x_length,
        }
    )


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
def find_registration_consensus(
    *,
    # Fractal parameters
    zarr_url: str,
    init_args: InitArgsRegistrationConsensus,
    # Core parameters
    roi_table: str = "FOV_ROI_table",
    # Advanced parameters
    new_roi_table: str | None = None,
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
    # Open all containers and load ROI tables, keeping containers for reuse
    # during the write phase. Reference acquisition is handled first so that
    # _find_roi_consensus receives its ROIs (zero-shift) as rois[0].
    ref_ome_zarr = open_ome_zarr_container(zarr_url)
    containers = {zarr_url: ref_ome_zarr}
    tables = {zarr_url: ref_ome_zarr.get_generic_roi_table(roi_table)}
    for acq_zarr_url in init_args.zarr_url_list:
        if acq_zarr_url == zarr_url:
            continue
        acq_ome_zarr = open_ome_zarr_container(acq_zarr_url)
        containers[acq_zarr_url] = acq_ome_zarr
        tables[acq_zarr_url] = acq_ome_zarr.get_generic_roi_table(roi_table)

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
        containers[acq_zarr_url].add_table(
            name=new_roi_table,
            table=registered_table,
            overwrite=True,
        )


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=find_registration_consensus,
        logger_name=logger.name,
    )

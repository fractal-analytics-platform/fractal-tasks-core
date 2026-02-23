# Copyright 2022-2026 (C) BioVisionCenter, University of Zurich
"""
Applies pre-calculated registration to images
"""

import logging
import os
import shutil
import time

from ngio import OmeZarrContainer, Roi, open_ome_zarr_container, open_ome_zarr_well
from pydantic import validate_call

logger = logging.getLogger("apply_registration_to_image")


def _get_ref_path_heuristic(path_list: list[str], path: str) -> str:
    """
    Pick the best-matching reference path from `path_list` for a given `path`.

    Matches by the suffix (everything after the first ``_`` in the path name).
    If no suffix match is found, falls back to the first sorted entry and logs
    a warning. Used when a well contains multiple images of the same
    acquisition (e.g. ``['0', '0_illum_corr']``) and we need to find the
    reference counterpart of a given image (e.g. ``'1_illum_corr'``).

    Args:
        path_list: Candidate reference image paths in the well.
        path: Current image path whose reference counterpart is sought.

    Returns:
        The best-matching path from `path_list`.
    """

    def _split_suffix(s: str) -> str:
        parts = s.split("_", 1)
        return parts[1] if len(parts) > 1 else ""

    suffix = _split_suffix(path)
    for p in sorted(path_list):
        if _split_suffix(p) == suffix:
            return p
    logger.warning(
        "No heuristic reference acquisition match found, defaulting to first "
        f"option {sorted(path_list)[0]}."
    )
    return sorted(path_list)[0]


def _write_registered_ngio_image(
    source_ome_zarr: OmeZarrContainer,
    new_zarr_url: str,
    roi_pairs: list[tuple[Roi, Roi]],
) -> OmeZarrContainer:
    """
    Write a registered OME-Zarr image to disk using pre-computed ROI pairs.

    Creates a new image container derived from the source (same shape, dtype,
    and metadata, initialised to zeros), writes image data from each acquisition
    ROI into the corresponding reference ROI position, then builds the pyramid
    using linear interpolation.

    Args:
        source_ome_zarr: Source image container (current acquisition).
        new_zarr_url: Path where the new registered image will be written.
        roi_pairs: List of (acq_roi, ref_roi) tuples. For each pair the data
            is read from `acq_roi` in the source and written to `ref_roi` in
            the new image. Regions not covered by any pair remain zero.

    Returns:
        The newly created OmeZarrContainer.
    """
    new_ome_zarr = source_ome_zarr.derive_image(new_zarr_url, overwrite=True)
    source_image = source_ome_zarr.get_image()
    new_image = new_ome_zarr.get_image()
    for acq_roi, ref_roi in roi_pairs:
        patch = source_image.get_roi(acq_roi)
        new_image.set_roi(ref_roi, patch)
    new_image.consolidate(order="linear")
    return new_ome_zarr


def _write_registered_ngio_label(
    acq_ome_zarr: OmeZarrContainer,
    new_ome_zarr: OmeZarrContainer,
    label_name: str,
    roi_pairs: list[tuple[Roi, Roi]],
) -> None:
    """
    Write a registered label image into an existing new OME-Zarr container.

    Derives an empty label from the source container, writes label data from
    each acquisition ROI into the reference ROI position, then builds the
    pyramid using nearest-neighbour interpolation (appropriate for integer
    segmentation masks).

    Args:
        acq_ome_zarr: Source image container (current acquisition).
        new_ome_zarr: Target container where the registered label is written.
        label_name: Name of the label to process.
        roi_pairs: List of (acq_roi, ref_roi) tuples (same as for the image).
    """
    acq_label = acq_ome_zarr.get_label(label_name)
    new_label = new_ome_zarr.derive_label(label_name, overwrite=True)
    for acq_roi, ref_roi in roi_pairs:
        patch = acq_label.get_roi(acq_roi)
        new_label.set_roi(ref_roi, patch)
    new_label.consolidate()


@validate_call
def apply_registration_to_image(
    *,
    # Fractal parameters
    zarr_url: str,
    # Core parameters
    registered_roi_table: str,
    reference_acquisition: int = 0,
    register_labels: bool = True,
    overwrite_input: bool = True,
):
    """
    Apply registration to images by using a registered ROI table

    This task consists of 4 parts:

    1. Mask all regions in images that are not available in the
    registered ROI table and store each acquisition aligned to the
    reference_acquisition (by looping over ROIs).
    2. Do the same for all label images.
    3. Copy all tables from the non-aligned image to the aligned image
    (currently only works well if the only tables are well & FOV ROI tables
    (registered and original). Not implemented for measurement tables and
    other ROI tables).
    4. Clean up: Delete the old, non-aligned image and rename the new,
    aligned image to take over its place.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        registered_roi_table: Name of the ROI table which has been registered
            and will be applied to mask and shift the images.
            Examples: `registered_FOV_ROI_table` => loop over the field of
            views, `registered_well_ROI_table` => process the whole well as
            one image.
        reference_acquisition: Which acquisition to register against. Uses the
            OME-NGFF HCS well metadata acquisition keys to find the reference
            acquisition.
        register_labels: Whether to also apply the registration to the label
            images. If True, all label images will be registered in the same
            way as the main image. If False, only the main image is registered.
        overwrite_input: Whether the old image data should be replaced with the
            newly registered image data. Currently only implemented for
            `overwrite_input=True`.

    """
    logger.info(
        f"Running `apply_registration_to_image` on {zarr_url=}, "
        f"{registered_roi_table=} and {reference_acquisition=}. "
        f"Using {overwrite_input=} and {register_labels=}"
    )

    well_url, old_img_path = zarr_url.rstrip("/").rsplit("/", 1)
    new_zarr_url = f"{well_url}/{old_img_path}_registered"

    # Resolve the zarr_url for the reference acquisition
    ome_zarr_well = open_ome_zarr_well(well_url)
    if reference_acquisition not in ome_zarr_well.acquisition_ids:
        raise ValueError(
            f"{reference_acquisition=} was not one of the available "
            f"acquisitions {ome_zarr_well.acquisition_ids} for well {well_url}"
        )
    ref_paths = ome_zarr_well.paths(reference_acquisition)
    if len(ref_paths) > 1:
        ref_path = _get_ref_path_heuristic(ref_paths, old_img_path)
        logger.warning(
            "Running registration when there are multiple images of the same "
            "acquisition in a well. Using a heuristic to match the reference "
            f"acquisition. Using {ref_path} as the reference image."
        )
    else:
        ref_path = ref_paths[0]
    reference_zarr_url = f"{well_url}/{ref_path}"

    # Open containers and load registered ROI tables
    ref_ome_zarr = open_ome_zarr_container(reference_zarr_url)
    acq_ome_zarr = open_ome_zarr_container(zarr_url)
    roi_table_ref = ref_ome_zarr.get_roi_table(registered_roi_table)
    roi_table_acq = acq_ome_zarr.get_roi_table(registered_roi_table)

    # Build ROI pairs by name (order-independent; names already validated by
    # find_registration_consensus)
    rois_ref = {roi.name: roi for roi in roi_table_ref.rois()}
    rois_acq = {roi.name: roi for roi in roi_table_acq.rois()}

    roi_pairs: list[tuple[Roi, Roi]] = []
    for name, roi_ref in rois_ref.items():
        if name not in rois_ref:
            raise ValueError(
                f"ROI with name {name} found in acquisition {zarr_url} but not "
                f"in reference acquisition {reference_zarr_url}."
            )
        if name not in rois_acq:
            raise ValueError(
                f"ROI with name {name} found in reference acquisition "
                f"{reference_zarr_url} but not in acquisition {zarr_url}."
            )
        roi_pairs.append((rois_acq[name], roi_ref))

    ####################
    # Process images
    ####################
    logger.info("Write the registered Zarr image to disk")
    new_ome_zarr = _write_registered_ngio_image(acq_ome_zarr, new_zarr_url, roi_pairs)

    ####################
    # Process labels
    ####################
    label_list = acq_ome_zarr.list_labels()
    if register_labels and label_list:
        logger.info(f"Processing the label images: {label_list}")
        for label_name in label_list:
            _write_registered_ngio_label(
                acq_ome_zarr, new_ome_zarr, label_name, roi_pairs
            )

    ####################
    # Copy tables
    # 1. Copy all ROI tables from the reference acquisition.
    # 2. Copy all non-ROI tables from the given acquisition.
    ####################
    ref_roi_table_names = set(ref_ome_zarr.list_tables(filter_types="roi_table"))
    acq_non_roi_table_names = set(acq_ome_zarr.list_tables()) - set(
        acq_ome_zarr.list_tables(filter_types="roi_table")
    )

    tables_to_copy: dict[str, OmeZarrContainer] = {
        name: ref_ome_zarr for name in ref_roi_table_names
    }
    for table_name in acq_non_roi_table_names:
        if reference_zarr_url != zarr_url:
            logger.warning(
                f"{zarr_url} contained a table that is not a ROI table. "
                "The `Apply Registration To Image task` is best used before "
                "additional tables are generated. It will copy "
                f"{table_name} from this acquisition without applying any "
                "transformations. This will work well if it contains "
                "measurements. But if it is a custom ROI table coming from "
                "another task, the transformation is not applied and it will "
                "not match with the registered image anymore."
            )
        tables_to_copy[table_name] = acq_ome_zarr

    if tables_to_copy:
        logger.info(f"Processing the tables: {list(tables_to_copy.keys())}")
        max_retries = 20
        sleep_time = 10
        for table_name, source in tables_to_copy.items():
            logger.info(f"Copying table: {table_name}")
            # Retry loop to guard against race conditions (see issue #516)
            for attempt in range(max_retries):
                try:
                    table = source.get_table(table_name)
                    new_ome_zarr.add_table(name=table_name, table=table, overwrite=True)
                    break
                except Exception:
                    logger.debug(
                        f"Table {table_name} not found in attempt {attempt}. "
                        f"Waiting {sleep_time} seconds before trying again."
                    )
                    time.sleep(sleep_time)
            else:
                raise RuntimeError(
                    f"Table {table_name} not found after {max_retries} attempts. "
                    "Check whether this table actually exists. If it does, "
                    "this may be a race condition issue."
                )

    ####################
    # Clean up Zarr file
    ####################
    if overwrite_input:
        logger.info("Replace original zarr image with the newly created Zarr image")
        # Potential for race conditions: Every acquisition reads the
        # reference acquisition, but the reference acquisition also gets
        # modified
        # See issue #516 for the details
        os.rename(zarr_url, f"{zarr_url}_tmp")
        os.rename(new_zarr_url, zarr_url)
        shutil.rmtree(f"{zarr_url}_tmp")
        image_list_updates = dict(image_list_updates=[dict(zarr_url=zarr_url)])
    else:
        image_list_updates = dict(
            image_list_updates=[dict(zarr_url=new_zarr_url, origin=zarr_url)]
        )
        # Update the well metadata to include the new image. We use
        # parallel_safe=True (the default) so that ngio uses a FileLock,
        # and atomic=True to ensure the metadata write is protected by that
        # lock (guards against the race condition of multiple acquisitions
        # modifying the well metadata simultaneously, see issue #516).
        new_img_path = f"{old_img_path}_registered"
        ome_zarr_well = open_ome_zarr_well(well_url)
        acq_id = ome_zarr_well.get_image_acquisition_id(old_img_path)
        ome_zarr_well.atomic_add_image(new_img_path, acquisition_id=acq_id, strict=True)

    return image_list_updates


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=apply_registration_to_image,
        logger_name=logger.name,
    )

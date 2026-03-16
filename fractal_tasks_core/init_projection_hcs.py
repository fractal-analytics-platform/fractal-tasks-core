# Copyright 2022-2026 (C) BioVisionCenter, University of Zurich
"""
Task that copies the structure of an OME-NGFF zarr array to a new one.
"""

import logging
from functools import cache
from pathlib import Path
from typing import Any

from ngio import OmeZarrPlate, OmeZarrWell, open_ome_zarr_plate, open_ome_zarr_well
from ngio.utils import NgioFileExistsError, NgioFileNotFoundError
from pydantic import Field, validate_call

from fractal_tasks_core._projection_utils import DaskProjectionMethod, InitArgsMIP
from fractal_tasks_core._utils import format_template_name

logger = logging.getLogger("init_projection_hcs")


@cache
def _open_well(well_path) -> OmeZarrWell:
    """
    Given the absolute `well_url` for an OME-Zarr plate,
        return the well object.
    """
    try:
        well = open_ome_zarr_well(well_path, mode="r", cache=True)
    except NgioFileNotFoundError:
        raise NgioFileNotFoundError(
            f"Could not open well {well_path}. "
            "Ensure that the path is correct and the file exists."
        )
    return well


def _get_plate(
    current_plate_url: str,
    proj_plate_url: str,
    re_initialize_plate: bool = False,
) -> OmeZarrPlate:
    """
    Given the absolute `plate_url` for an OME-Zarr plate,
        return the plate object.

    If the plate already exists, return it.
    If it does not exist, or if `re_initialize_plate` is True,
        create a proj plate and return it.
    """
    if re_initialize_plate or not Path(proj_plate_url).exists():
        logger.info(f"Creating proj plate: {proj_plate_url}")
        proj_plate_name = proj_plate_url.split("/")[-1]
        plate = open_ome_zarr_plate(current_plate_url).derive_plate(
            proj_plate_url,
            plate_name=proj_plate_name,
            overwrite=re_initialize_plate,
            keep_acquisitions=True,
        )
        logger.info(f"proj plate created: {plate}")
        return plate

    plate = open_ome_zarr_plate(proj_plate_url)
    logger.info(f"Plate already exists: {plate}")
    return plate


@validate_call
def init_projection_hcs(
    *,
    # Fractal parameters
    zarr_urls: list[str],
    zarr_dir: str,
    method: DaskProjectionMethod = DaskProjectionMethod.MIP,
    output_plate_name: str = Field(
        default="{plate_name}_{method}",
        pattern=r"^.*\{plate_name\}.*$",
    ),
    # Advanced parameters
    overwrite: bool = False,
    re_initialize_plate: bool = False,
) -> dict[str, Any]:
    """
    Duplicate the OME-Zarr HCS structure for a set of zarr_urls.

    This task only processes the zarr images in the zarr_urls, not all the
    images in the plate. It copies all the  plate & well structure, but none
    of the image metadata or the actual image data:

    - For each plate, create a new OME-Zarr HCS plate with the attributes for
        all the images in zarr_urls
    - For each well (in each plate), create a new zarr subgroup with the
       same attributes as the original one.

    Args:
        zarr_urls: List of paths or urls to the individual OME-Zarr image to
            be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        zarr_dir: path of the directory where the new OME-Zarrs will be
            created.
            (standard argument for Fractal tasks, managed by Fractal server).
            zarr_url: Path or url to the individual OME-Zarr image to be processed.
        method: Choose which method to use for intensity projection along the
            Z axis.
        output_plate_name: The template for the output plate name. To make sure
            that the output plate is unique it must contain the placeholder
            {plate_name}, and it can optionally contain the placeholder {method}.
        overwrite: If True, previous projected images with the same "output_plate_name"
            will be overwritten.
        re_initialize_plate: If True, the projection plate will be re-initialized
            even if it already exists. If False, the task will incrementally add the
            projected images to the existing plate if it already exists.

    Returns:
        Setup information required by the Compute Projection (HCS) task.
    """
    parallelization_list = []

    # A dictionary to store the plates and avoid re-initializing them multiple
    # times
    proj_plates: dict[str, OmeZarrPlate] = {}
    # A dictionary to store the images and avoid re-initializing querying all
    # wells multiple times
    proj_plates_images_paths: dict[str, list[str]] = {}

    # Generate parallelization list
    for zarr_url in zarr_urls:
        # Check if the zarr_url is valid
        if len(zarr_url.rstrip("/").split("/")) < 4:
            raise ValueError(
                f"Invalid zarr_url: {zarr_url}. "
                "The zarr_url of an image in a plate should be of the form "
                "`/path/to/plate_name/row/column/image_path`. "
                "The zarr_url given is too short to be valid."
            )
        *base, plate_name, row, column, image_path = zarr_url.rstrip("/").split("/")
        base_dir = "/".join(base)

        plate_url = f"{base_dir}/{plate_name}"
        plate_name = plate_name.rstrip(".zarr")  # Remove .zarr extension if present
        proj_plate_name = format_template_name(
            output_plate_name,
            plate_name=plate_name,
            method=method.abbreviation,
        )
        # Make sure the proj_plate_name ends with .zarr
        if not proj_plate_name.endswith(".zarr"):
            proj_plate_name = f"{proj_plate_name}.zarr"
        proj_plate_url = f"{zarr_dir}/{proj_plate_name}"

        if proj_plate_url not in proj_plates:
            _proj_plate = _get_plate(
                current_plate_url=plate_url,
                proj_plate_url=proj_plate_url,
                re_initialize_plate=re_initialize_plate,
            )
            proj_plates[proj_plate_url] = _proj_plate
            proj_plates_images_paths[proj_plate_url] = _proj_plate.images_paths()

        proj_plate = proj_plates[proj_plate_url]
        proj_plate_images_paths = proj_plates_images_paths[proj_plate_url]
        well_path = f"{plate_url}/{row}/{column}"
        well = _open_well(well_path)
        acquisition_id = well.get_image_acquisition_id(image_path)

        proj_image_path = f"{row}/{column}/{image_path}"

        if proj_image_path in proj_plate_images_paths:
            if not overwrite:
                raise NgioFileExistsError(
                    f"Image {proj_image_path} already exists in "
                    f"{proj_plate_url}. Set `overwrite=True` "
                    "to overwrite it."
                )
            logger.info(
                f"Image {proj_image_path} already exists in {proj_plate_url}. "
                "Overwriting it."
            )

        else:
            proj_plate.add_image(
                row=row,
                column=column,
                image_path=image_path,
                acquisition_id=acquisition_id,
            )
            proj_plates_images_paths[proj_plate_url].append(proj_image_path)

        proj_zarr_url = f"{proj_plate_url}/{proj_image_path}"
        proj_init = InitArgsMIP(
            origin_url=zarr_url,
            method=method,
            # Since we checked for existence above,
            # we can safely set this to True
            overwrite=True,
            new_plate_name=proj_plate_name,
        )
        parallelization_item = {
            "zarr_url": proj_zarr_url,
            "init_args": proj_init.model_dump(),
        }
        parallelization_list.append(parallelization_item)

    _open_well.cache_clear()
    return dict(parallelization_list=parallelization_list)


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=init_projection_hcs,
        logger_name=logger.name,
    )

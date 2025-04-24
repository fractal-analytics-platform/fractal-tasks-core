# Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Tommaso Comparin <tommaso.comparin@exact-lab.it>
# Marco Franzon <marco.franzon@exact-lab.it>
#
# This file is part of Fractal and was originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
"""
Task that copies the structure of an OME-NGFF zarr array to a new one.
"""
import logging
from functools import cache
from pathlib import Path
from typing import Any

from ngio import OmeZarrPlate
from ngio import OmeZarrWell
from ngio import open_ome_zarr_plate
from ngio import open_ome_zarr_well
from ngio.utils import NgioFileExistsError
from ngio.utils import NgioFileNotFoundError
from pydantic import validate_call

import fractal_tasks_core
from fractal_tasks_core.tasks.io_models import InitArgsMIP
from fractal_tasks_core.tasks.projection_utils import DaskProjectionMethod

logger = logging.getLogger(__name__)


__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__


@cache
def _open_well(well_path) -> OmeZarrWell:
    """
    Given the absolute `well_url` for an OME-Zarr plate,
        return the well object.
    """
    try:
        well = open_ome_zarr_well(
            well_path, mode="r", cache=True, parallel_safe=False
        )
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
            parallel_safe=False,
        )
        logger.info(f"proj plate created: {plate}")
        return plate

    plate = open_ome_zarr_plate(proj_plate_url, parallel_safe=False)
    logger.info(f"Plate already exists: {plate}")
    return plate


@validate_call
def copy_ome_zarr_hcs_plate(
    *,
    # Fractal parameters
    zarr_urls: list[str],
    zarr_dir: str,
    method: DaskProjectionMethod = DaskProjectionMethod.MIP,
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

    Note: this task makes use of methods from the `Attributes` class, see
    https://zarr.readthedocs.io/en/stable/api/attrs.html.

    Args:
        zarr_urls: List of paths or urls to the individual OME-Zarr image to
            be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        zarr_dir: path of the directory where the new OME-Zarrs will be
            created.
            (standard argument for Fractal tasks, managed by Fractal server).
        method: Choose which method to use for intensity projection along the
            Z axis. mip is the default and performs a maximum intensity
            projection. minip performs a minimum intensity projection, meanip
            a mean intensity projection and sumip a sum intensity projection.
        overwrite: If `True`, overwrite the MIP images if they are
            already present in the new OME-Zarr Plate.
        re_initialize_plate: If `True`, re-initialize the plate, deleting all
            existing wells and images. If `False`, the task will only
            incrementally add new wells and images to the plate.

    Returns:
        A parallelization list to be used in a compute task to fill the wells
        with OME-Zarr images.
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
        *base, plate_name, row, column, image_path = zarr_url.rstrip(
            "/"
        ).split("/")
        base_dir = "/".join(base)

        plate_url = f"{base_dir}/{plate_name}"
        proj_plate_name = (
            f"{plate_name}".rstrip(".zarr") + f"_{method.value}.zarr"
        )
        proj_plate_url = f"{zarr_dir}/{proj_plate_name}"

        if proj_plate_url not in proj_plates:
            _proj_plate = _get_plate(
                current_plate_url=plate_url,
                proj_plate_url=proj_plate_url,
                re_initialize_plate=re_initialize_plate,
            )
            proj_plates[proj_plate_url] = _proj_plate
            proj_plates_images_paths[
                proj_plate_url
            ] = _proj_plate.images_paths()

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
            method=method.value,
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
        task_function=copy_ome_zarr_hcs_plate,
        logger_name=logger.name,
    )

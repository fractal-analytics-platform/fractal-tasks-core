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
from typing import Any

from pydantic import validate_call

import fractal_tasks_core
from fractal_tasks_core.ngff.specs import NgffPlateMeta
from fractal_tasks_core.ngff.specs import WellInPlate
from fractal_tasks_core.ngff.zarr_utils import load_NgffPlateMeta
from fractal_tasks_core.ngff.zarr_utils import load_NgffWellMeta
from fractal_tasks_core.tasks.io_models import InitArgsMIP
from fractal_tasks_core.zarr_utils import open_zarr_group_with_overwrite

logger = logging.getLogger(__name__)


__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__


def _get_plate_url_from_image_url(zarr_url: str) -> str:
    """
    Given the absolute `zarr_url` for an OME-Zarr image within an HCS plate,
    return the path to the plate zarr group.
    """
    zarr_url = zarr_url.rstrip("/")
    plate_path = "/".join(zarr_url.split("/")[:-3])
    return plate_path


def _get_well_sub_url(zarr_url: str) -> str:
    """
    Given the absolute `zarr_url` for an OME-Zarr image within an HCS plate,
    return the path to the image zarr group.
    """
    zarr_url = zarr_url.rstrip("/")
    well_url = "/".join(zarr_url.split("/")[-3:-1])
    return well_url


def _get_image_sub_url(zarr_url: str) -> str:
    """
    Given the absolute `zarr_url` for an OME-Zarr image, return the image
    zarr-group name.
    """
    zarr_url = zarr_url.rstrip("/")
    image_sub_url = zarr_url.split("/")[-1]
    return image_sub_url


def _generate_wells_rows_columns(
    well_list: list[str],
) -> tuple[list[WellInPlate], list[str], list[str]]:
    """
    Generate the plate well metadata based on the list of wells.
    """
    rows = []
    columns = []
    wells = []
    for well in well_list:
        rows.append(well.split("/")[0])
        columns.append(well.split("/")[1])
    rows = sorted(list(set(rows)))
    columns = sorted(list(set(columns)))
    for well in well_list:
        wells.append(
            WellInPlate(
                path=well,
                rowIndex=rows.index(well.split("/")[0]),
                columnIndex=columns.index(well.split("/")[1]),
            )
        )

    return wells, rows, columns


def _generate_plate_well_metadata(
    zarr_urls: list[str],
) -> tuple[dict[str, dict], dict[str, dict[str, dict]], dict[str, dict]]:
    """
    Generate metadata for OME-Zarr HCS plates & wells.

    Based on the list of zarr_urls, generate metadata for all plates and all
    their wells.

    Args:
        zarr_urls: List of paths or urls to the individual OME-Zarr image to
            be processed.

    Returns:
        plate_metadata_dicts: Dictionary of plate plate metadata. The structure
            is: {"old_plate_name": NgffPlateMeta (as dict)}.
        new_well_image_attrs: Dictionary of image lists for the new wells.
            The structure is: {"old_plate_name": {"old_well_name":
            [ImageInWell(as dict)]}}
        well_image_attrs: Dictionary of Image attributes of the existing wells.
    """
    # TODO: Simplify this block. Currently complicated, because we need to loop
    # through all potential plates, all their wells & their images to build up
    # the metadata for the plate & well.
    plate_metadata_dicts = {}
    plate_wells = {}
    well_image_attrs = {}
    new_well_image_attrs = {}
    for zarr_url in zarr_urls:
        # Extract plate/well/image parts of `zarr_url`
        old_plate_url = _get_plate_url_from_image_url(zarr_url)
        well_sub_url = _get_well_sub_url(zarr_url)
        curr_img_sub_url = _get_image_sub_url(zarr_url)

        # The first time a plate is found, create its metadata
        if old_plate_url not in plate_metadata_dicts:
            logger.info(f"Reading plate metadata of {old_plate_url=}")
            old_plate_meta = load_NgffPlateMeta(old_plate_url)
            plate_metadata = dict(
                plate=dict(
                    acquisitions=old_plate_meta.plate.acquisitions,
                    field_count=old_plate_meta.plate.field_count,
                    name=old_plate_meta.plate.name,
                    # The new field count could be different from the old
                    # field count
                    version=old_plate_meta.plate.version,
                )
            )
            plate_metadata_dicts[old_plate_url] = plate_metadata
            plate_wells[old_plate_url] = []
            well_image_attrs[old_plate_url] = {}
            new_well_image_attrs[old_plate_url] = {}

        # The first time a plate/well pair is found, create the well metadata
        if well_sub_url not in plate_wells[old_plate_url]:
            plate_wells[old_plate_url].append(well_sub_url)
            old_well_url = f"{old_plate_url}/{well_sub_url}"
            logger.info(f"Reading well metadata of {old_well_url}")
            well_attrs = load_NgffWellMeta(old_well_url)
            well_image_attrs[old_plate_url][well_sub_url] = well_attrs.well
            new_well_image_attrs[old_plate_url][well_sub_url] = []

        # Find images of the current well with name matching the current image
        # TODO: clarify whether this list must always have length 1
        curr_well_image_list = [
            img
            for img in well_image_attrs[old_plate_url][well_sub_url].images
            if img.path == curr_img_sub_url
        ]
        new_well_image_attrs[old_plate_url][
            well_sub_url
        ] += curr_well_image_list

    # Fill in the plate metadata based on all available wells
    for old_plate_url in plate_metadata_dicts:
        well_list, row_list, column_list = _generate_wells_rows_columns(
            plate_wells[old_plate_url]
        )
        plate_metadata_dicts[old_plate_url]["plate"]["columns"] = []
        for column in column_list:
            plate_metadata_dicts[old_plate_url]["plate"]["columns"].append(
                {"name": column}
            )

        plate_metadata_dicts[old_plate_url]["plate"]["rows"] = []
        for row in row_list:
            plate_metadata_dicts[old_plate_url]["plate"]["rows"].append(
                {"name": row}
            )
        plate_metadata_dicts[old_plate_url]["plate"]["wells"] = well_list

        # Validate with NgffPlateMeta model
        plate_metadata_dicts[old_plate_url] = NgffPlateMeta(
            **plate_metadata_dicts[old_plate_url]
        ).model_dump(exclude_none=True)

    return plate_metadata_dicts, new_well_image_attrs, well_image_attrs


@validate_call
def copy_ome_zarr_hcs_plate(
    *,
    # Fractal parameters
    zarr_urls: list[str],
    zarr_dir: str,
    # Advanced parameters
    suffix: str = "mip",
    overwrite: bool = False,
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
        suffix: The suffix that is used to transform `plate.zarr` into
            `plate_suffix.zarr`. Note that `None` is not currently supported.
        overwrite: If `True`, overwrite the task output.

    Returns:
        A parallelization list to be used in a compute task to fill the wells
        with OME-Zarr images.
    """

    # Preliminary check
    if suffix is None or suffix == "":
        raise ValueError(
            "Running copy_ome_zarr_hcs_plate without a suffix would lead to"
            "overwriting of the existing HCS plates."
        )

    parallelization_list = []

    # Generate parallelization list
    for zarr_url in zarr_urls:
        old_plate_url = _get_plate_url_from_image_url(zarr_url)
        well_sub_url = _get_well_sub_url(zarr_url)
        old_plate_name = old_plate_url.split(".zarr")[-2].split("/")[-1]
        new_plate_name = f"{old_plate_name}_{suffix}"
        zarrurl_plate_new = f"{zarr_dir}/{new_plate_name}.zarr"
        curr_img_sub_url = _get_image_sub_url(zarr_url)
        new_zarr_url = f"{zarrurl_plate_new}/{well_sub_url}/{curr_img_sub_url}"
        parallelization_item = dict(
            zarr_url=new_zarr_url,
            init_args=dict(origin_url=zarr_url),
        )
        InitArgsMIP(**parallelization_item["init_args"])
        parallelization_list.append(parallelization_item)

    # Generate the plate metadata & parallelization list
    (
        plate_attrs_dicts,
        new_well_image_attrs,
        well_image_attrs,
    ) = _generate_plate_well_metadata(zarr_urls=zarr_urls)

    # Create the new OME-Zarr HCS plate
    for old_plate_url, plate_attrs in plate_attrs_dicts.items():
        old_plate_name = old_plate_url.split(".zarr")[-2].split("/")[-1]
        new_plate_name = f"{old_plate_name}_{suffix}"
        zarrurl_new = f"{zarr_dir}/{new_plate_name}.zarr"
        logger.info(f"{old_plate_url=}")
        logger.info(f"{zarrurl_new=}")
        new_plate_group = open_zarr_group_with_overwrite(
            zarrurl_new, overwrite=overwrite
        )
        new_plate_group.attrs.put(plate_attrs)

        # Write well groups:
        for well_sub_url in new_well_image_attrs[old_plate_url]:
            new_well_group = new_plate_group.create_group(f"{well_sub_url}")
            well_attrs = dict(
                well=dict(
                    images=[
                        img.model_dump(exclude_none=True)
                        for img in new_well_image_attrs[old_plate_url][
                            well_sub_url
                        ]
                    ],
                    version=well_image_attrs[old_plate_url][
                        well_sub_url
                    ].version,
                )
            )
            new_well_group.attrs.put(well_attrs)

    return dict(parallelization_list=parallelization_list)


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=copy_ome_zarr_hcs_plate,
        logger_name=logger.name,
    )

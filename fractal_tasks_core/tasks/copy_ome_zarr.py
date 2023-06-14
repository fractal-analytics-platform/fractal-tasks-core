"""
Copyright 2022 (C)
    Friedrich Miescher Institute for Biomedical Research and
    University of Zurich

    Original authors:
    Tommaso Comparin <tommaso.comparin@exact-lab.it>
    Marco Franzon <marco.franzon@exact-lab.it>

    This file is part of Fractal and was originally developed by eXact lab
    S.r.l.  <exact-lab.it> under contract with Liberali Lab from the Friedrich
    Miescher Institute for Biomedical Research and Pelkmans Lab from the
    University of Zurich.

Task that copies the structure of an OME-NGFF zarr array to a new one
"""
import logging
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from typing import Sequence

import anndata as ad
import zarr
from anndata._io.specs import write_elem
from pydantic.decorator import validate_arguments

import fractal_tasks_core
from fractal_tasks_core.lib_regions_of_interest import (
    convert_ROIs_from_3D_to_2D,
)
from fractal_tasks_core.lib_zattrs_utils import extract_zyx_pixel_sizes

logger = logging.getLogger(__name__)


__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__


@validate_arguments
def copy_ome_zarr(
    *,
    input_paths: Sequence[str],
    output_path: str,
    metadata: Dict[str, Any],
    project_to_2D: bool = True,
    suffix: str = "mip",
    ROI_table_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:

    """
    Duplicate an input zarr structure to a new path.

    This task copies all the structure, but none of the image data:

        1. For each plate, create a new zarr group with the same attributes as
           the original one.
        2. For each well (in each plate), create a new zarr subgroup with the
           same attributes as the original one.
        3. For each image (in each well), create a new zarr subgroup with the
           same attributes as the original one.
        4. For each image (in each well), copy the relevant AnnData tables from
           the original source.

    Note: this task makes use of methods from the ``Attributes`` class, see
    https://zarr.readthedocs.io/en/stable/api/attrs.html.

    :param input_paths: List of input paths where the image data is stored
                        as OME-Zarrs. Should point to the parent folder
                        containing one or many OME-Zarr files, not the
                        actual OME-Zarr file.
                        Example: ["/some/path/"]
                        This task only supports a single input path.
                        (standard argument for Fractal tasks,
                        managed by Fractal server)
    :param output_path: Path were the output of this task is stored.
                        Example: "/some/path/" => puts the new OME-Zarr file
                        in the same folder as the input OME-Zarr file
                        "/some/new_path" => puts the new OME-Zarr file
                        into a new folder at ``/some/new_path``
                        (standard argument for Fractal tasks,
                        managed by Fractal server)
    :param metadata: dictionary containing metadata about the OME-Zarr.
                     This task requires the following elements to be present
                     in the metadata:
                     "plate": List of plates. Example: ["MyPlate.zarr"]
                     "well": List of wells in the OME-Zarr plate.
                     ["MyPlate.zarr/B/03", "MyPlate.zarr/B/05"]
                     "image": List of images in the OME-Zarr plate. Example:
                     ["MyPlate.zarr/B/03/0", "MyPlate.zarr/B/05/0"]
                     (standard argument for Fractal tasks,
                     managed by Fractal server)
    :param project_to_2D: If ``True``, apply a 3D->2D projection to the ROI
                          tables that are copied to the new OME-Zarr.
    :param suffix: The suffix that is used to transform ``plate.zarr`` into
                   ``plate_suffix.zarr``. Note that `None` is not currently
                   supported.
    :param ROI_table_names: List of Anndata table names to be copied.
                            If ``None``, it is replaced by ``["FOV_ROI_table",
                            "well_ROI_table"]``. Note: copying non-ROI tables
                            may fail if ``project_to_2D=True``.
    :return: An update to the metadata table with new "plate", "well",
             "image" entries (now with the suffix in the plate name).
    """

    # Preliminary check
    if len(input_paths) > 1:
        raise NotImplementedError
    if suffix is None:
        # FIXME create a standard suffix (with timestamp)
        raise NotImplementedError

    if ROI_table_names is None:
        # FIXME: are these defaults OK?
        ROI_table_names = ["FOV_ROI_table", "well_ROI_table"]

    # List all plates
    in_path = Path(input_paths[0])
    list_plates = [p.as_posix() for p in Path(in_path).glob("*.zarr")]
    logger.info(f"{list_plates=}")

    meta_update: Dict[str, Any] = {"copy_ome_zarr": {}}
    meta_update["copy_ome_zarr"]["suffix"] = suffix
    meta_update["copy_ome_zarr"]["sources"] = {}

    # Loop over all plates
    for zarrurl_old in list_plates:
        zarrfile = zarrurl_old.split("/")[-1]
        old_plate_name = zarrfile.split(".zarr")[0]
        new_plate_name = f"{old_plate_name}_{suffix}"
        new_plate_dir = Path(output_path).resolve()
        zarrurl_new = f"{(new_plate_dir / new_plate_name).as_posix()}.zarr"
        meta_update["copy_ome_zarr"]["sources"][new_plate_name] = zarrurl_old

        logger.info(f"{zarrurl_old=}")
        logger.info(f"{zarrurl_new=}")
        logger.info(f"{meta_update=}")

        # Replicate plate attrs
        old_plate_group = zarr.open_group(zarrurl_old, mode="r")
        new_plate_group = zarr.open(zarrurl_new)
        new_plate_group.attrs.put(old_plate_group.attrs.asdict())

        well_paths = [
            well["path"] for well in new_plate_group.attrs["plate"]["wells"]
        ]
        logger.info(f"{well_paths=}")
        for well_path in well_paths:

            # Replicate well attrs
            old_well_group = zarr.open_group(
                f"{zarrurl_old}/{well_path}", mode="r"
            )
            new_well_group = zarr.group(f"{zarrurl_new}/{well_path}")
            new_well_group.attrs.put(old_well_group.attrs.asdict())

            image_paths = [
                image["path"]
                for image in new_well_group.attrs["well"]["images"]
            ]
            logger.info(f"{image_paths=}")

            for image_path in image_paths:

                # Replicate image attrs
                old_image_group = zarr.open_group(
                    f"{zarrurl_old}/{well_path}/{image_path}", mode="r"
                )
                new_image_group = zarr.group(
                    f"{zarrurl_new}/{well_path}/{image_path}"
                )
                new_image_group.attrs.put(old_image_group.attrs.asdict())

                # Extract pixel sizes, if needed
                if ROI_table_names:

                    new_tables_group = new_image_group.create_group("tables/")
                    new_tables_group.attrs["tables"] = ROI_table_names
                    if project_to_2D:
                        path_FOV_zattrs = (
                            f"{zarrurl_old}/{well_path}/{image_path}/.zattrs"
                        )
                        pxl_sizes_zyx = extract_zyx_pixel_sizes(
                            path_FOV_zattrs, level=0
                        )
                        pxl_size_z = pxl_sizes_zyx[0]

                    # Copy the tables in ROI_table_names
                    for ROI_table_name in ROI_table_names:

                        logger.info(
                            f"I will now read {ROI_table_name} from "
                            f"{zarrurl_old=}, convert it to 2D, and "
                            "write it back to the new zarr file."
                        )
                        ROI_table = ad.read_zarr(
                            f"{zarrurl_old}/{well_path}/{image_path}/"
                            f"tables/{ROI_table_name}"
                        )
                        # Convert 3D FOVs to 2D
                        if project_to_2D:
                            new_ROI_table = convert_ROIs_from_3D_to_2D(
                                ROI_table, pxl_size_z
                            )
                        # Write new table
                        write_elem(
                            new_tables_group, ROI_table_name, new_ROI_table
                        )

    for key in ["plate", "well", "image"]:
        meta_update[key] = [
            component.replace(".zarr", f"_{suffix}.zarr")
            for component in metadata[key]
        ]

    return meta_update


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=copy_ome_zarr,
        logger_name=logger.name,
    )

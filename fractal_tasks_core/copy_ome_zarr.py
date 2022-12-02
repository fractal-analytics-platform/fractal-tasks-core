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
from anndata.experimental import write_elem

import fractal_tasks_core
from fractal_tasks_core.lib_regions_of_interest import (
    convert_ROIs_from_3D_to_2D,
)
from fractal_tasks_core.lib_zattrs_utils import extract_zyx_pixel_sizes

logger = logging.getLogger(__name__)


__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__


def copy_ome_zarr(
    *,
    input_paths: Sequence[Path],
    output_path: Path,
    metadata: Dict[str, Any],
    project_to_2D: bool = True,
    suffix: Optional[str] = None,
    ROI_table_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:

    """
    Duplicate an input zarr structure to a new path.

    More detailed description:

    1. For each plate, create a new zarr group with the same attributes as the
    original one.
    2. For each well (in each plate), create a new zarr subgroup with the same
    attributes as the original one.
    3. For each image (in each well), create a new zarr subgroup with the same
    attributes as the original one.
    4. For each image (in each well), copy the relevant AnnData tables from the
    original source.

    Note: this task makes use of methods from the ``Attributes`` class, see
    https://zarr.readthedocs.io/en/stable/api/attrs.html.

    Examples of some inputs::
      input_paths = ["/tmp/out/*.zarr"]
      output_path = "/tmp/out_mip/*.zarr"

    :param input_paths: TBD
    :param output_path: TBD
    :param metadata: TBD
    :param project_to_2D: If ``True``, apply a 3D->2D projection to the ROI
                          tables that are copied to the new zarr.
    :param suffix: The suffix that is used to transform ``plate.zarr`` into
                   ``plate_suffix.zarr``. Note that `None` is not currently
                   supported.

    :param ROI_table_names: List of ROI-table names to be copied. If ``None``,
                            it is replaced by ``["FOV_ROI_table",
                            "well_ROI_table"]``. Note: copying non-ROI tables
                            may fail if ``project_to_2D=True``.
    """

    # Preliminary check
    if len(input_paths) > 1:
        raise NotImplementedError
    if suffix is None:
        # FIXME create a standard suffix (with timestamp)
        raise NotImplementedError

    if ROI_table_names is None:
        ROI_table_names = ["FOV_ROI_table", "well_ROI_table"]

    # List all plates
    in_path = input_paths[0]
    list_plates = [
        p.as_posix() for p in in_path.parent.resolve().glob(in_path.name)
    ]
    logger.info(f"{list_plates=}")

    meta_update: Dict[str, Any] = {"copy_zarr": {}}
    meta_update["copy_zarr"]["suffix"] = suffix
    meta_update["copy_zarr"]["sources"] = {}

    # Loop over all plates
    for zarrurl_old in list_plates:
        zarrfile = zarrurl_old.split("/")[-1]
        old_plate_name = zarrfile.split(".zarr")[0]
        new_plate_name = f"{old_plate_name}_{suffix}"
        new_plate_dir = output_path.resolve().parent
        zarrurl_new = f"{(new_plate_dir / new_plate_name).as_posix()}.zarr"
        meta_update["copy_zarr"]["sources"][new_plate_name] = zarrurl_old

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

                    group_tables = new_image_group.create_group("tables/")
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
                        ROI_table = ad.read_zarr(
                            f"{zarrurl_old}/{well_path}/{image_path}/"
                            f"tables/{ROI_table_name}"
                        )
                        # Convert 3D FOVs to 2D
                        if project_to_2D:
                            ROI_table = convert_ROIs_from_3D_to_2D(
                                ROI_table, pxl_size_z
                            )
                        # Write new table
                        write_elem(group_tables, ROI_table_name, ROI_table)

    for key in ["plate", "well", "image"]:
        meta_update[key] = [
            component.replace(".zarr", f"_{suffix}.zarr")
            for component in metadata[key]
        ]

    return meta_update


if __name__ == "__main__":
    from pydantic import BaseModel
    from fractal_tasks_core._utils import run_fractal_task

    class TaskArguments(BaseModel):
        input_paths: Sequence[Path]
        output_path: Path
        metadata: Dict[str, Any]
        project_to_2D: bool = True
        suffix: Optional[str] = None
        ROI_table_names: Optional[Sequence[str]] = None

    run_fractal_task(
        task_function=copy_ome_zarr,
        TaskArgsModel=TaskArguments,
        logger_name=logger.name,
    )

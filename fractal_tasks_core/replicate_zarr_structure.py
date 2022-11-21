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
import json
import logging
from glob import glob
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


def replicate_zarr_structure(
    *,
    input_paths: Sequence[Path],
    output_path: Path,
    metadata: Dict[str, Any],
    project_to_2D: bool = True,
    suffix: str = None,
) -> Dict[str, Any]:

    """
    Duplicate an input zarr structure to a new path.
    If project_to_2D=True, adapt it to host a maximum-intensity projection
    (that is, with a single Z layer).

    Examples
      input_paths[0] = /tmp/out/*.zarr    (Path)
      output_path = /tmp/out_mip/*.zarr   (Path)

    :param input_paths: TBD
    :param output_path: TBD
    :param metadata: TBD
    :param project_to_2D: TBD
    :param suffix: TBD
    """

    # Preliminary check
    if len(input_paths) > 1:
        raise NotImplementedError
    if suffix is None:
        # FIXME create a standard suffix (with timestamp)
        raise NotImplementedError

    # List all plates
    in_path = input_paths[0]
    list_plates = [
        p.as_posix() for p in in_path.parent.resolve().glob(in_path.name)
    ]
    logger.info("{list_plates=}")

    meta_update: Dict[str, Any] = {"replicate_zarr": {}}
    meta_update["replicate_zarr"]["suffix"] = suffix
    meta_update["replicate_zarr"]["sources"] = {}

    # Loop over all plates
    for zarrurl_old in list_plates:
        zarrfile = zarrurl_old.split("/")[-1]
        old_plate_name = zarrfile.split(".zarr")[0]
        new_plate_name = f"{old_plate_name}_{suffix}"
        new_plate_dir = output_path.resolve().parent
        zarrurl_new = f"{(new_plate_dir / new_plate_name).as_posix()}.zarr"
        meta_update["replicate_zarr"]["sources"][new_plate_name] = zarrurl_old

        logger.info(f"{zarrurl_old=}")
        logger.info(f"{zarrurl_new=}")
        logger.info(f"{meta_update=}")

        # Identify properties of input zarr file
        well_rows_columns = sorted(
            [rc.split("/")[-2:] for rc in glob(zarrurl_old + "/*/*")]
        )
        row_list = [
            well_row_column[0] for well_row_column in well_rows_columns
        ]
        col_list = [
            well_row_column[1] for well_row_column in well_rows_columns
        ]
        row_list = sorted(list(set(row_list)))
        col_list = sorted(list(set(col_list)))

        group_plate = zarr.group(zarrurl_new)
        plate = zarrurl_old.replace(".zarr", "").split("/")[-1]
        logger.info(f"{plate=}")
        group_plate.attrs["plate"] = {
            "acquisitions": [{"id": 0, "name": plate}],
            "columns": [{"name": col} for col in col_list],
            "rows": [{"name": row} for row in row_list],
            "wells": [
                {
                    "path": well_row_column[0] + "/" + well_row_column[1],
                    "rowIndex": row_list.index(well_row_column[0]),
                    "columnIndex": col_list.index(well_row_column[1]),
                }
                for well_row_column in well_rows_columns
            ],
        }

        for row, column in well_rows_columns:

            # Find FOVs in COL/ROW/.zattrs
            path_well_zattrs = f"{zarrurl_old}/{row}/{column}/.zattrs"
            with open(path_well_zattrs) as well_zattrs_file:
                well_zattrs = json.load(well_zattrs_file)
            well_images = well_zattrs["well"]["images"]
            list_FOVs = sorted([img["path"] for img in well_images])

            # Create well group
            group_well = group_plate.create_group(f"{row}/{column}/")
            group_well.attrs["well"] = {
                "images": well_images,
                "version": __OME_NGFF_VERSION__,
            }

            # Check that only the 0-th FOV exists
            FOV = 0
            if len(list_FOVs) > 1:
                raise Exception(
                    "ERROR: we are in a single-merged-FOV scheme, "
                    f"but there are {len(list_FOVs)} FOVs."
                )

            # Create FOV group
            group_FOV = group_well.create_group(f"{FOV}/")

            # Copy .zattrs file at the COL/ROW/FOV level
            path_FOV_zattrs = f"{zarrurl_old}/{row}/{column}/{FOV}/.zattrs"
            with open(path_FOV_zattrs) as FOV_zattrs_file:
                FOV_zattrs = json.load(FOV_zattrs_file)
            for key in FOV_zattrs.keys():
                group_FOV.attrs[key] = FOV_zattrs[key]

            # Read FOV ROI table
            FOV_ROI_table = ad.read_zarr(
                f"{zarrurl_old}/{row}/{column}/0/tables/FOV_ROI_table"
            )
            well_ROI_table = ad.read_zarr(
                f"{zarrurl_old}/{row}/{column}/0/tables/well_ROI_table"
            )

            # Convert 3D FOVs to 2D
            if project_to_2D:
                # Read pixel sizes from zattrs file
                pxl_sizes_zyx = extract_zyx_pixel_sizes(
                    path_FOV_zattrs, level=0
                )
                pxl_size_z = pxl_sizes_zyx[0]
                FOV_ROI_table = convert_ROIs_from_3D_to_2D(
                    FOV_ROI_table, pxl_size_z
                )
                well_ROI_table = convert_ROIs_from_3D_to_2D(
                    well_ROI_table, pxl_size_z
                )

            # Create table group and write new table
            group_tables = group_FOV.create_group("tables/")
            write_elem(group_tables, "FOV_ROI_table", FOV_ROI_table)
            write_elem(group_tables, "well_ROI_table", well_ROI_table)

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

    run_fractal_task(
        task_function=replicate_zarr_structure,
        TaskArgsModel=TaskArguments,
        logger_name=logger.name,
    )

"""
Copyright 2022 (C)
    Friedrich Miescher Institute for Biomedical Research and
    University of Zurich

    Original authors:
    Tommaso Comparin <tommaso.comparin@exact-lab.it>

    This file is part of Fractal and was originally developed by eXact lab
    S.r.l.  <exact-lab.it> under contract with Liberali Lab from the Friedrich
    Miescher Institute for Biomedical Research and Pelkmans Lab from the
    University of Zurich.

Create structure for OME-NGFF zarr array
"""
import os
from glob import glob
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence

import pandas as pd
import zarr
from anndata.experimental import write_elem

import fractal_tasks_core
from fractal_tasks_core.lib_omero import define_omero_channels
from fractal_tasks_core.lib_parse_filename_metadata import parse_filename
from fractal_tasks_core.lib_regions_of_interest import prepare_FOV_ROI_table
from fractal_tasks_core.lib_regions_of_interest import prepare_well_ROI_table
from fractal_tasks_core.metadata_parsing import parse_yokogawa_metadata


__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__

import logging

logger = logging.getLogger(__name__)


def create_zarr_structure(
    *,
    input_paths: Sequence[Path],
    output_path: Path,
    metadata: Dict[str, Any] = None,
    channel_parameters: Dict[str, Any],
    num_levels: int = 2,
    coarsening_xy: int = 2,
    metadata_table: str = "mrf_mlf",
) -> Dict[str, Any]:
    """
    Create a OME-NGFF zarr folder, without reading/writing image data

    Find plates (for each folder in input_paths)
        * glob image files
        * parse metadata from image filename to identify plates
        * identify populated channels

    Create a zarr folder (for each plate)
        * parse mlf metadata
        * identify wells and field of view (FOV)
        * create FOV ZARR
        * verify that channels are uniform (i.e., same channels)

    :param input_paths: TBD (common to all tasks)
    :param output_path: TBD (common to all tasks)
    :param metadata: TBD (common to all tasks)
    :param channel_parameters: TBD
    :param num_levels: number of resolution-pyramid levels
    :param coarsening_xy: linear coarsening factor between subsequent levels
    :param metadata_table: TBD
    """

    # Preliminary checks on metadata_table
    if metadata_table != "mrf_mlf" and not isinstance(
        metadata_table, pd.core.frame.DataFrame
    ):
        raise Exception(
            "ERROR: metadata_table must be a known string or a "
            "pandas DataFrame}"
        )
    if metadata_table != "mrf_mlf":
        raise NotImplementedError(
            "We currently only support "
            'metadata_table="mrf_mlf", '
            f"and not {metadata_table}"
        )
    if channel_parameters is None:
        raise Exception(
            "Missing channel_parameters argument in " "create_zarr_structure"
        )

    # Identify all plates and all channels, across all input folders
    plates = []
    channels = None
    dict_plate_paths = {}
    dict_plate_prefixes: Dict[str, Any] = {}

    # FIXME
    # find a smart way to remove it
    ext_glob_pattern = input_paths[0].name

    for in_path in input_paths:
        input_filename_iter = in_path.parent.glob(in_path.name)

        tmp_channels = []
        tmp_plates = []
        for fn in input_filename_iter:
            try:
                filename_metadata = parse_filename(fn.name)
                plate_prefix = filename_metadata["plate_prefix"]
                plate = filename_metadata["plate"]
                if plate not in dict_plate_prefixes.keys():
                    dict_plate_prefixes[plate] = plate_prefix
                tmp_plates.append(plate)
                tmp_channels.append(
                    f"A{filename_metadata['A']}_C{filename_metadata['C']}"
                )
            except IndexError:
                logger.info("IndexError for ", fn)
                pass
        tmp_plates = sorted(list(set(tmp_plates)))
        tmp_channels = sorted(list(set(tmp_channels)))

        info = (
            f"Listing all plates/channels from {in_path.as_posix()}\n"
            f"Plates:   {tmp_plates}\n"
            f"Channels: {tmp_channels}\n"
        )

        # Check that only one plate is found
        if len(tmp_plates) > 1:
            raise Exception(f"{info}ERROR: {len(tmp_plates)} plates detected")
        elif len(tmp_plates) == 0:
            raise Exception(f"{info}ERROR: No plates detected")
        plate = tmp_plates[0]

        # If plate already exists in other folder, add suffix
        if plate in plates:
            ind = 1
            new_plate = f"{plate}_{ind}"
            while new_plate in plates:
                new_plate = f"{plate}_{ind}"
                ind += 1
            logger.info(
                f"WARNING: {plate} already exists, renaming it as {new_plate}"
            )
            plates.append(new_plate)
            dict_plate_prefixes[new_plate] = dict_plate_prefixes[plate]
            plate = new_plate
        else:
            plates.append(plate)

        # Check that channels are the same as in previous plates
        if channels is None:
            channels = tmp_channels[:]
        else:
            if channels != tmp_channels:
                raise Exception(
                    f"ERROR\n{info}\nERROR: expected channels {channels}"
                )

        # Update dict_plate_paths
        dict_plate_paths[plate] = in_path.parent

    # Check that all channels are in the allowed_channels
    if not set(channels).issubset(set(channel_parameters.keys())):
        msg = "ERROR in create_zarr_structure\n"
        msg += f"channels: {channels}\n"
        msg += f"allowed_channels: {channel_parameters.keys()}\n"
        raise Exception(msg)

    # Create actual_channels, i.e. a list of entries like "A01_C01"
    actual_channels = []
    for ind_ch, ch in enumerate(channels):
        actual_channels.append(ch)
    logger.info(f"actual_channels: {actual_channels}")

    # Clean up dictionary channel_parameters

    zarrurls: Dict[str, List[str]] = {"plate": [], "well": [], "image": []}

    ################################################################
    for plate in plates:
        # Define plate zarr
        zarrurl = f"{plate}.zarr"
        in_path = dict_plate_paths[plate]
        logger.info(f"Creating {zarrurl}")
        group_plate = zarr.group(output_path.parent / zarrurl)
        zarrurls["plate"].append(zarrurl)

        # Obtain FOV-metadata dataframe
        try:
            # FIXME
            # Find a smart way to include these metadata files in the dataset
            # e.g., as resources
            if metadata_table == "mrf_mlf":
                mrf_path = f"{in_path}/MeasurementDetail.mrf"
                mlf_path = f"{in_path}/MeasurementData.mlf"
                site_metadata, total_files = parse_yokogawa_metadata(
                    mrf_path, mlf_path
                )
                has_mrf_mlf_metadata = True

                # Extract pixel sizes and bit_depth
                pixel_size_z = site_metadata["pixel_size_z"][0]
                pixel_size_y = site_metadata["pixel_size_y"][0]
                pixel_size_x = site_metadata["pixel_size_x"][0]
                bit_depth = site_metadata["bit_depth"][0]
        except FileNotFoundError:
            logger.info("Missing metadata files")
            has_mrf_mlf_metadata = False
            pixel_size_x = pixel_size_y = pixel_size_z = 1

        if min(pixel_size_z, pixel_size_y, pixel_size_x) < 1e-9:
            raise Exception(pixel_size_z, pixel_size_y, pixel_size_x)

        # Identify all wells
        plate_prefix = dict_plate_prefixes[plate]

        plate_image_iter = glob(f"{in_path}/{plate_prefix}_{ext_glob_pattern}")

        wells = [
            parse_filename(os.path.basename(fn))["well"]
            for fn in plate_image_iter
        ]
        wells = sorted(list(set(wells)))

        # Verify that all wells have all channels
        for well in wells:
            well_image_iter = glob(
                f"{in_path}/{plate_prefix}_{well}{ext_glob_pattern}"
            )
            well_channels = []
            for fpath in well_image_iter:
                try:
                    filename_metadata = parse_filename(os.path.basename(fpath))
                    well_channels.append(
                        f"A{filename_metadata['A']}_C{filename_metadata['C']}"
                    )
                except IndexError:
                    logger.info(f"Skipping {fpath}")
            well_channels = sorted(list(set(well_channels)))
            if well_channels != actual_channels:
                raise Exception(
                    f"ERROR: well {well} in plate {plate} (prefix: "
                    f"{plate_prefix}) has missing channels.\n"
                    f"Expected: {actual_channels}\n"
                    f"Found: {well_channels}.\n"
                )

        well_rows_columns = [
            ind for ind in sorted([(n[0], n[1:]) for n in wells])
        ]
        row_list = [
            well_row_column[0] for well_row_column in well_rows_columns
        ]
        col_list = [
            well_row_column[1] for well_row_column in well_rows_columns
        ]
        row_list = sorted(list(set(row_list)))
        col_list = sorted(list(set(col_list)))

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

            group_well = group_plate.create_group(f"{row}/{column}/")

            group_well.attrs["well"] = {
                "images": [{"path": "0"}],
                "version": __OME_NGFF_VERSION__,
            }

            group_FOV = group_well.create_group("0/")  # noqa: F841
            zarrurls["well"].append(f"{plate}.zarr/{row}/{column}/")
            zarrurls["image"].append(f"{plate}.zarr/{row}/{column}/0/")

            group_FOV.attrs["multiscales"] = [
                {
                    "version": __OME_NGFF_VERSION__,
                    "axes": [
                        {"name": "c", "type": "channel"},
                        {
                            "name": "z",
                            "type": "space",
                            "unit": "micrometer",
                        },
                        {
                            "name": "y",
                            "type": "space",
                            "unit": "micrometer",
                        },
                        {
                            "name": "x",
                            "type": "space",
                            "unit": "micrometer",
                        },
                    ],
                    "datasets": [
                        {
                            "path": f"{ind_level}",
                            "coordinateTransformations": [
                                {
                                    "type": "scale",
                                    "scale": [
                                        pixel_size_z,
                                        pixel_size_y
                                        * coarsening_xy**ind_level,
                                        pixel_size_x
                                        * coarsening_xy**ind_level,
                                    ],
                                }
                            ],
                        }
                        for ind_level in range(num_levels)
                    ],
                }
            ]

            group_FOV.attrs["omero"] = {
                "id": 1,  # FIXME does this depend on the plate number?
                "name": "TBD",
                "version": __OME_NGFF_VERSION__,
                "channels": define_omero_channels(
                    actual_channels, channel_parameters, bit_depth
                ),
            }

            # FIXME
            if has_mrf_mlf_metadata:
                group_tables = group_FOV.create_group("tables/")  # noqa: F841

                # Prepare FOV/well tables
                FOV_ROIs_table = prepare_FOV_ROI_table(
                    site_metadata.loc[f"{row+column}"],
                )
                # Prepare and write anndata table of well ROIs
                well_ROIs_table = prepare_well_ROI_table(
                    site_metadata.loc[f"{row+column}"],
                )
                # Write tables
                write_elem(group_tables, "FOV_ROI_table", FOV_ROIs_table)
                write_elem(group_tables, "well_ROI_table", well_ROIs_table)

    metadata_update = dict(
        plate=zarrurls["plate"],
        well=zarrurls["well"],
        image=zarrurls["image"],
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        channel_list=actual_channels,
        original_paths=[str(p) for p in input_paths],
    )
    return metadata_update


if __name__ == "__main__":
    from pydantic import BaseModel
    from fractal_tasks_core._utils import run_fractal_task

    class TaskArguments(BaseModel):
        input_paths: Sequence[Path]
        output_path: Path
        metadata: Optional[Dict[str, Any]]
        channel_parameters: Dict[str, Any]
        num_levels: int = 2
        coarsening_xy: int = 2
        metadata_table: str = "mrf_mlf"

    run_fractal_task(
        task_function=create_zarr_structure,
        TaskArgsModel=TaskArguments,
        logger_name=logger.name,
    )

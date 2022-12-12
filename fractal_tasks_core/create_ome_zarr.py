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
from typing import Sequence

import pandas as pd
import zarr
from anndata.experimental import write_elem

import fractal_tasks_core
from fractal_tasks_core.lib_channels import check_well_channel_labels
from fractal_tasks_core.lib_channels import define_omero_channels
from fractal_tasks_core.lib_channels import validate_allowed_channel_input
from fractal_tasks_core.lib_metadata_parsing import parse_yokogawa_metadata
from fractal_tasks_core.lib_parse_filename_metadata import parse_filename
from fractal_tasks_core.lib_regions_of_interest import prepare_FOV_ROI_table
from fractal_tasks_core.lib_regions_of_interest import prepare_well_ROI_table
from fractal_tasks_core.lib_remove_FOV_overlaps import remove_FOV_overlaps


__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__

import logging

logger = logging.getLogger(__name__)


def create_ome_zarr(
    *,
    input_paths: Sequence[Path],
    output_path: Path,
    metadata: Dict[str, Any],
    allowed_channels: Sequence[Dict[str, Any]],
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
    :param num_levels: Number of resolution-pyramid levels
    :param coarsening_xy: Linear coarsening factor between subsequent levels
    :param allowed_channels: A list of channel dictionaries, where each channel
                             must include the ``wavelength_id`` key and where
                             the corresponding values should be unique across
                             channels.
    :param metadata_table: If equal to ``"mrf_mlf"``, parse Yokogawa metadata
                           from mrf/mlf files in the input_path folder; else,
                           the full path to a csv file containing
                           the parsed metadata table.
    """

    # Preliminary checks on metadata_table
    if metadata_table != "mrf_mlf" and not metadata_table.endswith(".csv"):
        raise ValueError(
            "metadata_table must be a known string or a "
            "csv file containing a pandas dataframe"
        )
    if metadata_table.endswith(".csv") and not os.path.isfile(metadata_table):
        raise FileNotFoundError(f"Missing file: {metadata_table=}")

    # Identify all plates and all channels, across all input folders
    plates = []
    actual_wavelength_ids = None
    dict_plate_paths = {}
    dict_plate_prefixes: Dict[str, Any] = {}

    # Preliminary checks on allowed_channels argument
    validate_allowed_channel_input(allowed_channels)

    # FIXME
    # find a smart way to remove it
    ext_glob_pattern = input_paths[0].name

    for in_path in input_paths:
        input_filename_iter = in_path.parent.glob(in_path.name)

        tmp_wavelength_ids = []
        tmp_plates = []
        for fn in input_filename_iter:
            try:
                filename_metadata = parse_filename(fn.name)
                plate_prefix = filename_metadata["plate_prefix"]
                plate = filename_metadata["plate"]
                if plate not in dict_plate_prefixes.keys():
                    dict_plate_prefixes[plate] = plate_prefix
                tmp_plates.append(plate)
                A = filename_metadata["A"]
                C = filename_metadata["C"]
                tmp_wavelength_ids.append(f"A{A}_C{C}")
            except ValueError as e:
                logger.warning(
                    f'Skipping "{fn.name}". Original error: ' + str(e)
                )
        tmp_plates = sorted(list(set(tmp_plates)))
        tmp_wavelength_ids = sorted(list(set(tmp_wavelength_ids)))

        info = (
            f"Listing all plates/channels from {in_path.as_posix()}\n"
            f"Plates:   {tmp_plates}\n"
            f"Channels: {tmp_wavelength_ids}\n"
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
        if actual_wavelength_ids is None:
            actual_wavelength_ids = tmp_wavelength_ids[:]
        else:
            if actual_wavelength_ids != tmp_wavelength_ids:
                raise Exception(
                    f"ERROR\n{info}\nERROR:"
                    f" expected channels {actual_wavelength_ids}"
                )

        # Update dict_plate_paths
        dict_plate_paths[plate] = in_path.parent

    # Check that all channels are in the allowed_channels
    allowed_wavelength_ids = [
        channel["wavelength_id"] for channel in allowed_channels
    ]
    if not set(actual_wavelength_ids).issubset(set(allowed_wavelength_ids)):
        msg = "ERROR in create_ome_zarr\n"
        msg += f"actual_wavelength_ids: {actual_wavelength_ids}\n"
        msg += f"allowed_wavelength_ids: {allowed_wavelength_ids}\n"
        raise Exception(msg)

    # Create actual_channels, i.e. a list of the channel dictionaries which are
    # present
    actual_channels = [
        channel
        for channel in allowed_channels
        if channel["wavelength_id"] in actual_wavelength_ids
    ]

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

        if metadata_table == "mrf_mlf":
            mrf_path = f"{in_path}/MeasurementDetail.mrf"
            mlf_path = f"{in_path}/MeasurementData.mlf"
            site_metadata, total_files = parse_yokogawa_metadata(
                mrf_path, mlf_path
            )
            site_metadata = remove_FOV_overlaps(site_metadata)

        # If a metadata table was passed, load it and use it directly
        elif metadata_table.endswith(".csv"):
            site_metadata = pd.read_csv(metadata_table)
            site_metadata.set_index(["well_id", "FieldIndex"], inplace=True)

        # Extract pixel sizes and bit_depth
        pixel_size_z = site_metadata["pixel_size_z"][0]
        pixel_size_y = site_metadata["pixel_size_y"][0]
        pixel_size_x = site_metadata["pixel_size_x"][0]
        bit_depth = site_metadata["bit_depth"][0]

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
            well_wavelength_ids = []
            for fpath in well_image_iter:
                try:
                    filename_metadata = parse_filename(os.path.basename(fpath))
                    well_wavelength_ids.append(
                        f"A{filename_metadata['A']}_C{filename_metadata['C']}"
                    )
                except IndexError:
                    logger.info(f"Skipping {fpath}")
            well_wavelength_ids = sorted(list(set(well_wavelength_ids)))
            if well_wavelength_ids != actual_wavelength_ids:
                raise Exception(
                    f"ERROR: well {well} in plate {plate} (prefix: "
                    f"{plate_prefix}) has missing channels.\n"
                    f"Expected: {actual_channels}\n"
                    f"Found: {well_wavelength_ids}.\n"
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

            group_image = group_well.create_group("0/")  # noqa: F841
            zarrurls["well"].append(f"{plate}.zarr/{row}/{column}/")
            zarrurls["image"].append(f"{plate}.zarr/{row}/{column}/0/")

            group_image.attrs["multiscales"] = [
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

            group_image.attrs["omero"] = {
                "id": 1,  # FIXME does this depend on the plate number?
                "name": "TBD",
                "version": __OME_NGFF_VERSION__,
                "channels": define_omero_channels(
                    channels=actual_channels, bit_depth=bit_depth
                ),
            }

            # Create tables zarr group for ROI tables
            group_tables = group_image.create_group("tables/")  # noqa: F841
            well_id = row + column

            # Prepare AnnData tables for FOV/well ROIs
            FOV_ROIs_table = prepare_FOV_ROI_table(site_metadata.loc[well_id])
            well_ROIs_table = prepare_well_ROI_table(
                site_metadata.loc[well_id]
            )

            # Write AnnData tables in the tables zarr group
            write_elem(group_tables, "FOV_ROI_table", FOV_ROIs_table)
            write_elem(group_tables, "well_ROI_table", well_ROIs_table)

    # Check that the different images in each well have unique channel labels.
    # Since we currently merge all fields of view in the same image, this check
    # is useless. It should remain there to catch an error in case we switch
    # back to one-image-per-field-of-view mode
    for well_path in zarrurls["well"]:
        check_well_channel_labels(
            well_zarr_path=str(output_path.parent / well_path)
        )

    metadata_update = dict(
        plate=zarrurls["plate"],
        well=zarrurls["well"],
        image=zarrurls["image"],
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        original_paths=[str(p) for p in input_paths],
    )
    return metadata_update


if __name__ == "__main__":
    from pydantic import BaseModel
    from fractal_tasks_core._utils import run_fractal_task

    class TaskArguments(BaseModel):
        input_paths: Sequence[Path]
        output_path: Path
        metadata: Dict[str, Any]
        allowed_channels: Sequence[Dict[str, Any]]
        num_levels: int = 2
        coarsening_xy: int = 2
        metadata_table: str = "mrf_mlf"

    run_fractal_task(
        task_function=create_ome_zarr,
        TaskArgsModel=TaskArguments,
        logger_name=logger.name,
    )

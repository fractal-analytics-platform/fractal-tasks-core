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

Create OME-NGFF zarr group, for multiplexing dataset
"""
import os
from glob import glob
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Literal
from typing import Sequence
from typing import Union

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


def create_ome_zarr_multiplex(
    *,
    input_paths: Sequence[Path],
    output_path: Path,
    metadata: Dict[str, Any],
    allowed_channels: Dict[str, Sequence[Dict[str, Any]]],
    num_levels: int = 2,
    coarsening_xy: int = 2,
    metadata_table: Union[Literal["mrf_mlf"], Dict[str, str]] = "mrf_mlf",
) -> Dict[str, Any]:
    """
    Create OME-NGFF structure and metadata to host a multiplexing dataset

    This task takes a set of image folders (i.e. different acquisition cycles)
    and build the internal structure and metadata of a OME-NGFF zarr group,
    without actually loading/writing the image data.

    Each input_paths should be treated as a different acquisition

    :param input_paths: list of image folders for different acquisition
                        cycles, e.g. in the form ``["/path/cycle1/*.png",
                        "/path/cycle2/*.png"]``
    :param output_path: parent folder for the output path, e.g.
                        ``"/outputpath/*.zarr"``
    :param metadata: standard fractal argument, not used in this task
    :param allowed_channels: TBD
    :param num_levels: number of resolution-pyramid levels
    :param coarsening_xy: Linear coarsening factor between subsequent levels
    :param metadata_table: If equal to ``"mrf_mlf"``, parse Yokogawa metadata
                           from mrf/mlf files in the ``input_paths`` folders;
                           else, a dictionary of key-value pairs like
                           ``(acquisition, path)`` with ``acquisition`` a
                           string and ``path`` pointing to a csv file
                           containing the parsed metadata table.
    """

    # Preliminary checks on metadata_table
    if metadata_table != "mrf_mlf" and not isinstance(metadata_table, Dict):
        raise ValueError(
            "ERROR: metadata_table must be a known string or a "
            "dict of csv file containing a pandas dataframe."
            f"The metadata_table provided was {metadata_table}"
        )
    elif isinstance(metadata_table, Dict):
        # Check that acquisition keys are string
        for key, value in metadata_table.items():
            if not isinstance(key, str):
                raise ValueError(f"{metadata_table=} has non-string keys")

        # Checks on the dict:
        # 1. Acquisitions as keys (same as keys of allowed_channels)
        # 2. Files end with ".csv"
        # 3. Files exist.
        if set(allowed_channels.keys()) != set(metadata_table.keys()):
            raise ValueError(
                "Mismatch in acquisition keys between "
                f"{allowed_channels.keys()=} and {metadata_table.keys()=}"
            )
        if not all([x.endswith(".csv") for x in metadata_table.values()]):
            raise ValueError(
                f"Some files in {metadata_table=} do not end with csv."
            )
        if not all([os.path.isfile(x) for x in metadata_table.values()]):
            raise ValueError(f"Some files in {metadata_table=} do not exist.")

    # Preliminary checks on allowed_channels
    # Note that in metadata the keys of dictionary arguments should be
    # strings (and not integers), so that they can be read from a JSON file
    for key, value in allowed_channels.items():
        if not isinstance(key, str):
            raise ValueError(f"{allowed_channels=} has non-string keys")
        validate_allowed_channel_input(value)

    # Identify all plates and all channels, per input folders
    dict_acquisitions: Dict = {}

    ext_glob_pattern = input_paths[0].name

    for ind_in_path, in_path in enumerate(input_paths):
        acquisition = str(ind_in_path)
        dict_acquisitions[acquisition] = {}

        actual_wavelength_ids = []
        plates = []
        plate_prefixes = []

        # Loop over all images
        input_filename_iter = in_path.parent.glob(ext_glob_pattern)
        for fn in input_filename_iter:
            try:
                filename_metadata = parse_filename(fn.name)
                plate = filename_metadata["plate"]
                plates.append(plate)
                plate_prefix = filename_metadata["plate_prefix"]
                plate_prefixes.append(plate_prefix)
                A = filename_metadata["A"]
                C = filename_metadata["C"]
                actual_wavelength_ids.append(f"A{A}_C{C}")
            except ValueError as e:
                logger.warning(
                    f'Skipping "{fn.name}". Original error: ' + str(e)
                )
        plates = sorted(list(set(plates)))
        actual_wavelength_ids = sorted(list(set(actual_wavelength_ids)))

        info = (
            f"Listing all plates/channels from {in_path.as_posix()}\n"
            f"Plates:   {plates}\n"
            f"Actual wavelength IDs: {actual_wavelength_ids}\n"
        )

        # Check that a folder includes a single plate
        if len(plates) > 1:
            raise ValueError(f"{info}ERROR: {len(plates)} plates detected")
        elif len(plates) == 0:
            raise ValueError(f"{info}ERROR: No plates detected")
        original_plate = plates[0]
        plate_prefix = plate_prefixes[0]

        # Replace plate with the one of acquisition 0, if needed
        if int(acquisition) > 0:
            plate = dict_acquisitions["0"]["plate"]
            logger.warning(
                f"For {acquisition=}, we replace {original_plate=} with "
                f"{plate=} (the one for acquisition 0)"
            )

        # Check that all channels are in the allowed_channels
        allowed_wavelength_ids = [
            c["wavelength_id"] for c in allowed_channels[acquisition]
        ]
        if not set(actual_wavelength_ids).issubset(
            set(allowed_wavelength_ids)
        ):
            msg = "ERROR in create_ome_zarr\n"
            msg += f"actual_wavelength_ids: {actual_wavelength_ids}\n"
            msg += f"allowed_wavelength_ids: {allowed_wavelength_ids}\n"
            raise ValueError(msg)

        # Create actual_channels, i.e. a list of the channel dictionaries which
        # are present
        actual_channels = [
            channel
            for channel in allowed_channels[acquisition]
            if channel["wavelength_id"] in actual_wavelength_ids
        ]

        logger.info(f"plate: {plate}")
        logger.info(f"actual_channels: {actual_channels}")

        dict_acquisitions[acquisition] = {}
        dict_acquisitions[acquisition]["plate"] = plate
        dict_acquisitions[acquisition]["original_plate"] = original_plate
        dict_acquisitions[acquisition]["plate_prefix"] = plate_prefix
        dict_acquisitions[acquisition]["image_folder"] = str(in_path.parent)
        dict_acquisitions[acquisition]["original_paths"] = [str(in_path)]
        dict_acquisitions[acquisition]["actual_channels"] = actual_channels

    acquisitions = sorted(list(dict_acquisitions.keys()))
    current_plates = [item["plate"] for item in dict_acquisitions.values()]
    if len(set(current_plates)) > 1:
        raise ValueError(f"{current_plates=}")
    plate = current_plates[0]

    zarrurl = dict_acquisitions[acquisitions[0]]["plate"] + ".zarr"
    full_zarrurl = str(output_path.parent / zarrurl)
    logger.info(f"Creating {full_zarrurl=}")
    group_plate = zarr.group(full_zarrurl)
    group_plate.attrs["plate"] = {
        "acquisitions": [
            {
                "id": int(acquisition),
                "name": dict_acquisitions[acquisition]["original_plate"],
            }
            for acquisition in acquisitions
        ]
    }

    zarrurls: Dict[str, List[str]] = {"well": [], "image": []}
    zarrurls["plate"] = [plate]

    ################################################################
    logging.info(f"{acquisitions=}")

    for acquisition in acquisitions:

        # Define plate zarr
        image_folder = dict_acquisitions[acquisition]["image_folder"]
        logger.info(f"Looking at {image_folder=}")

        # Obtain FOV-metadata dataframe
        if metadata_table == "mrf_mlf":
            mrf_path = f"{image_folder}/MeasurementDetail.mrf"
            mlf_path = f"{image_folder}/MeasurementData.mlf"
            site_metadata, total_files = parse_yokogawa_metadata(
                mrf_path, mlf_path
            )
            site_metadata = remove_FOV_overlaps(site_metadata)

        elif isinstance(metadata_table, Dict):
            site_metadata = pd.read_csv(metadata_table[acquisition])
            site_metadata.set_index(["well_id", "FieldIndex"], inplace=True)

        # Extract pixel sizes and bit_depth
        pixel_size_z = site_metadata["pixel_size_z"][0]
        pixel_size_y = site_metadata["pixel_size_y"][0]
        pixel_size_x = site_metadata["pixel_size_x"][0]
        bit_depth = site_metadata["bit_depth"][0]

        if min(pixel_size_z, pixel_size_y, pixel_size_x) < 1e-9:
            raise Exception(pixel_size_z, pixel_size_y, pixel_size_x)

        # Identify all wells
        plate_prefix = dict_acquisitions[acquisition]["plate_prefix"]
        glob_string = f"{image_folder}/{plate_prefix}_{ext_glob_pattern}"
        logger.info(f"{glob_string=}")
        plate_image_iter = glob(glob_string)

        wells = [
            parse_filename(os.path.basename(fn))["well"]
            for fn in plate_image_iter
        ]
        wells = sorted(list(set(wells)))
        logger.info(f"{wells=}")

        # Verify that all wells have all channels
        actual_channels = dict_acquisitions[acquisition]["actual_channels"]
        for well in wells:
            well_image_iter = glob(
                f"{image_folder}/{plate_prefix}_{well}{ext_glob_pattern}"
            )
            well_wavelength_ids = []
            for fpath in well_image_iter:
                try:
                    filename_metadata = parse_filename(os.path.basename(fpath))
                    A = filename_metadata["A"]
                    C = filename_metadata["C"]
                    well_wavelength_ids.append(f"A{A}_C{C}")
                except IndexError:
                    logger.info(f"Skipping {fpath}")
            well_wavelength_ids = sorted(list(set(well_wavelength_ids)))
            if well_wavelength_ids != actual_wavelength_ids:
                raise Exception(
                    f"ERROR: well {well} in plate {plate} (prefix: "
                    f"{plate_prefix}) has missing channels.\n"
                    f"Expected: {actual_wavelength_ids}\n"
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

        plate_attrs = group_plate.attrs["plate"]
        plate_attrs["columns"] = [{"name": col} for col in col_list]
        plate_attrs["rows"] = [{"name": row} for row in row_list]
        plate_attrs["wells"] = [
            {
                "path": well_row_column[0] + "/" + well_row_column[1],
                "rowIndex": row_list.index(well_row_column[0]),
                "columnIndex": col_list.index(well_row_column[1]),
            }
            for well_row_column in well_rows_columns
        ]
        group_plate.attrs["plate"] = plate_attrs

        for row, column in well_rows_columns:

            from zarr.errors import ContainsGroupError

            try:
                group_well = group_plate.create_group(f"{row}/{column}/")
                logging.info(f"Created new group_well at {row}/{column}/")
                group_well.attrs["well"] = {
                    "images": [
                        {
                            "path": f"{acquisition}",
                            "acquisition": int(acquisition),
                        }
                    ],
                    "version": __OME_NGFF_VERSION__,
                }
                zarrurls["well"].append(f"{plate}.zarr/{row}/{column}")
            except ContainsGroupError:
                group_well = zarr.open_group(
                    f"{full_zarrurl}/{row}/{column}/", mode="r+"
                )
                logging.info(
                    f"Loaded group_well from {full_zarrurl}/{row}/{column}"
                )
                current_images = group_well.attrs["well"]["images"] + [
                    {"path": f"{acquisition}", "acquisition": int(acquisition)}
                ]
                group_well.attrs["well"] = dict(
                    images=current_images,
                    version=group_well.attrs["well"]["version"],
                )

            group_image = group_well.create_group(
                f"{acquisition}/"
            )  # noqa: F841
            logging.info(f"Created image group {row}/{column}/{acquisition}")
            image = f"{plate}.zarr/{row}/{column}/{acquisition}"
            zarrurls["image"].append(image)

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
                    channels=actual_channels,
                    bit_depth=bit_depth,
                    label_prefix=acquisition,
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

    # Check that the different images (e.g. different cycles) in the each well
    # have unique labels
    for well_path in zarrurls["well"]:
        check_well_channel_labels(
            well_zarr_path=str(output_path.parent / well_path)
        )

    original_paths = {
        acquisition: dict_acquisitions[acquisition]["original_paths"]
        for acquisition in acquisitions
    }

    metadata_update = dict(
        plate=zarrurls["plate"],
        well=zarrurls["well"],
        image=zarrurls["image"],
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        original_paths=original_paths,
    )
    return metadata_update


if __name__ == "__main__":
    from pydantic import BaseModel
    from fractal_tasks_core._utils import run_fractal_task

    class TaskArguments(BaseModel):
        input_paths: Sequence[Path]
        output_path: Path
        metadata: Dict[str, Any]
        allowed_channels: Dict[str, Sequence[Dict[str, Any]]]
        num_levels: int = 2
        coarsening_xy: int = 2
        metadata_table: Union[Literal["mrf_mlf"], Dict[str, str]] = "mrf_mlf"

    run_fractal_task(
        task_function=create_ome_zarr_multiplex,
        TaskArgsModel=TaskArguments,
        logger_name=logger.name,
    )

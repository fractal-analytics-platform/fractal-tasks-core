# Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Tommaso Comparin <tommaso.comparin@exact-lab.it>
#
# This file is part of Fractal and was originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
"""
Create OME-NGFF zarr group, for multiplexing dataset.
"""
import os
from pathlib import Path
from typing import Any
from typing import Optional
from typing import Sequence

import pandas as pd
import zarr
from pydantic.decorator import validate_arguments
from zarr.errors import ContainsGroupError

import fractal_tasks_core
from fractal_tasks_core.cellvoyager.filenames import (
    glob_with_multiple_patterns,
)
from fractal_tasks_core.cellvoyager.filenames import parse_filename
from fractal_tasks_core.cellvoyager.metadata import (
    parse_yokogawa_metadata,
)
from fractal_tasks_core.channels import check_unique_wavelength_ids
from fractal_tasks_core.channels import check_well_channel_labels
from fractal_tasks_core.channels import define_omero_channels
from fractal_tasks_core.channels import OmeroChannel
from fractal_tasks_core.roi import prepare_FOV_ROI_table
from fractal_tasks_core.roi import prepare_well_ROI_table
from fractal_tasks_core.roi import remove_FOV_overlaps
from fractal_tasks_core.tables import write_table
from fractal_tasks_core.zarr_utils import open_zarr_group_with_overwrite


__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__

import logging

logger = logging.getLogger(__name__)


@validate_arguments
def create_ome_zarr_multiplex(
    *,
    input_paths: Sequence[str],
    output_path: str,
    metadata: dict[str, Any],
    allowed_channels: dict[str, list[OmeroChannel]],
    image_glob_patterns: Optional[list[str]] = None,
    num_levels: int = 5,
    coarsening_xy: int = 2,
    image_extension: str = "tif",
    metadata_table_files: Optional[dict[str, str]] = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """
    Create OME-NGFF structure and metadata to host a multiplexing dataset.

    This task takes a set of image folders (i.e. different acquisition cycles)
    and build the internal structure and metadata of a OME-NGFF zarr group,
    without actually loading/writing the image data.

    Each element in input_paths should be treated as a different acquisition.

    Args:
        input_paths: List of input paths where the image data from the
            microscope is stored (as TIF or PNG).  Each element of the list is
            treated as another cycle of the multiplexing data, the cycles are
            ordered by their order in this list.  Should point to the parent
            folder containing the images and the metadata files
            `MeasurementData.mlf` and `MeasurementDetail.mrf` (if present).
            Example: `["/path/cycle1/", "/path/cycle2/"]`. (standard argument
            for Fractal tasks, managed by Fractal server).
        output_path: Path were the output of this task is stored.
            Example: `"/some/path/"` => puts the new OME-Zarr file in the
            `/some/path/`.
            (standard argument for Fractal tasks, managed by Fractal server).
        metadata: This parameter is not used by this task.
            (standard argument for Fractal tasks, managed by Fractal server).
        allowed_channels: A dictionary of lists of `OmeroChannel`s, where
            each channel must include the `wavelength_id` attribute and where
            the `wavelength_id` values must be unique across each list.
            Dictionary keys represent channel indices (`"0","1",..`).
        image_glob_patterns: If specified, only parse images with filenames
            that match with all these patterns. Patterns must be defined as in
            https://docs.python.org/3/library/fnmatch.html, Example:
            `image_glob_pattern=["*_B03_*"]` => only process well B03
            `image_glob_pattern=["*_C09_*", "*F016*", "*Z[0-5][0-9]C*"]` =>
            only process well C09, field of view 16 and Z planes 0-59.
        num_levels: Number of resolution-pyramid levels. If set to `5`, there
            will be the full-resolution level and 4 levels of downsampled
            images.
        coarsening_xy: Linear coarsening factor between subsequent levels.
            If set to `2`, level 1 is 2x downsampled, level 2 is 4x downsampled
            etc.
        image_extension: Filename extension of images
            (e.g. `"tif"` or `"png"`).
        metadata_table_files: If `None`, parse Yokogawa metadata from mrf/mlf
            files in the input_path folder; else, a dictionary of key-value
            pairs like `(acquisition, path)` with `acquisition` a string
            and `path` pointing to a csv file containing the parsed metadata
            table.
        overwrite: If `True`, overwrite the task output.

    Returns:
        A metadata dictionary containing important metadata about the OME-Zarr
            plate, the images and some parameters required by downstream tasks
            (like `num_levels`).
    """

    if metadata_table_files:

        # Checks on the dict:
        # 1. Acquisitions as keys (same as keys of allowed_channels)
        # 2. Files end with ".csv"
        # 3. Files exist.
        if set(allowed_channels.keys()) != set(metadata_table_files.keys()):
            raise ValueError(
                "Mismatch in acquisition keys between "
                f"{allowed_channels.keys()=} and "
                f"{metadata_table_files.keys()=}"
            )
        for f in metadata_table_files.values():
            if not f.endswith(".csv"):
                raise ValueError(
                    f"{f} (in metadata_table_file) is not a csv file."
                )
            if not os.path.isfile(f):
                raise ValueError(
                    f"{f} (in metadata_table_file) does not exist."
                )

    # Preliminary checks on allowed_channels
    # Note that in metadata the keys of dictionary arguments should be
    # strings (and not integers), so that they can be read from a JSON file
    for key, _channels in allowed_channels.items():
        if not isinstance(key, str):
            raise ValueError(f"{allowed_channels=} has non-string keys")
        check_unique_wavelength_ids(_channels)

    # Identify all plates and all channels, per input folders
    dict_acquisitions: dict = {}

    for ind_in_path, in_path_str in enumerate(input_paths):
        acquisition = str(ind_in_path)
        in_path = Path(in_path_str)
        dict_acquisitions[acquisition] = {}

        actual_wavelength_ids = []
        plates = []
        plate_prefixes = []

        # Loop over all images
        patterns = [f"*.{image_extension}"]
        if image_glob_patterns:
            patterns.extend(image_glob_patterns)
        input_filenames = glob_with_multiple_patterns(
            folder=in_path_str,
            patterns=patterns,
        )
        for fn in input_filenames:
            try:
                filename_metadata = parse_filename(Path(fn).name)
                plate = filename_metadata["plate"]
                plates.append(plate)
                plate_prefix = filename_metadata["plate_prefix"]
                plate_prefixes.append(plate_prefix)
                A = filename_metadata["A"]
                C = filename_metadata["C"]
                actual_wavelength_ids.append(f"A{A}_C{C}")
            except ValueError as e:
                logger.warning(
                    f'Skipping "{Path(fn).name}". Original error: ' + str(e)
                )
        plates = sorted(list(set(plates)))
        actual_wavelength_ids = sorted(list(set(actual_wavelength_ids)))

        info = (
            "Listing all plates/channels:\n"
            f"Patterns: {patterns}\n"
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
            c.wavelength_id for c in allowed_channels[acquisition]
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
            if channel.wavelength_id in actual_wavelength_ids
        ]

        logger.info(f"plate: {plate}")
        logger.info(f"actual_channels: {actual_channels}")

        dict_acquisitions[acquisition] = {}
        dict_acquisitions[acquisition]["plate"] = plate
        dict_acquisitions[acquisition]["original_plate"] = original_plate
        dict_acquisitions[acquisition]["plate_prefix"] = plate_prefix
        dict_acquisitions[acquisition]["image_folder"] = in_path
        dict_acquisitions[acquisition]["original_paths"] = [in_path]
        dict_acquisitions[acquisition]["actual_channels"] = actual_channels
        dict_acquisitions[acquisition][
            "actual_wavelength_ids"
        ] = actual_wavelength_ids

    acquisitions = sorted(list(dict_acquisitions.keys()))
    current_plates = [item["plate"] for item in dict_acquisitions.values()]
    if len(set(current_plates)) > 1:
        raise ValueError(f"{current_plates=}")
    plate = current_plates[0]

    zarrurl = dict_acquisitions[acquisitions[0]]["plate"] + ".zarr"
    full_zarrurl = str(Path(output_path) / zarrurl)
    logger.info(f"Creating {full_zarrurl=}")
    # Call zarr.open_group wrapper, which handles overwrite=True/False
    group_plate = open_zarr_group_with_overwrite(
        full_zarrurl, overwrite=overwrite
    )
    group_plate.attrs["plate"] = {
        "acquisitions": [
            {
                "id": int(acquisition),
                "name": dict_acquisitions[acquisition]["original_plate"],
            }
            for acquisition in acquisitions
        ]
    }

    zarrurls: dict[str, list[str]] = {"well": [], "image": []}
    zarrurls["plate"] = [f"{plate}.zarr"]

    ################################################################
    logging.info(f"{acquisitions=}")

    for acquisition in acquisitions:

        # Define plate zarr
        image_folder = dict_acquisitions[acquisition]["image_folder"]
        logger.info(f"Looking at {image_folder=}")

        # Obtain FOV-metadata dataframe
        if metadata_table_files is None:
            mrf_path = f"{image_folder}/MeasurementDetail.mrf"
            mlf_path = f"{image_folder}/MeasurementData.mlf"
            site_metadata, total_files = parse_yokogawa_metadata(
                mrf_path, mlf_path, filename_patterns=image_glob_patterns
            )
            site_metadata = remove_FOV_overlaps(site_metadata)
        else:
            site_metadata = pd.read_csv(metadata_table_files[acquisition])
            site_metadata.set_index(["well_id", "FieldIndex"], inplace=True)

        # Extract pixel sizes and bit_depth
        pixel_size_z = site_metadata["pixel_size_z"][0]
        pixel_size_y = site_metadata["pixel_size_y"][0]
        pixel_size_x = site_metadata["pixel_size_x"][0]
        bit_depth = site_metadata["bit_depth"][0]

        if min(pixel_size_z, pixel_size_y, pixel_size_x) < 1e-9:
            raise ValueError(pixel_size_z, pixel_size_y, pixel_size_x)

        # Identify all wells
        plate_prefix = dict_acquisitions[acquisition]["plate_prefix"]
        patterns = [f"{plate_prefix}_*.{image_extension}"]
        if image_glob_patterns:
            patterns.extend(image_glob_patterns)
        plate_images = glob_with_multiple_patterns(
            folder=str(image_folder),
            patterns=patterns,
        )

        wells = [
            parse_filename(os.path.basename(fn))["well"] for fn in plate_images
        ]
        wells = sorted(list(set(wells)))
        logger.info(f"{wells=}")

        # Verify that all wells have all channels
        actual_channels = dict_acquisitions[acquisition]["actual_channels"]
        for well in wells:
            patterns = [f"{plate_prefix}_{well}_*.{image_extension}"]
            if image_glob_patterns:
                patterns.extend(image_glob_patterns)
            well_images = glob_with_multiple_patterns(
                folder=str(image_folder),
                patterns=patterns,
            )

            well_wavelength_ids = []
            for fpath in well_images:
                try:
                    filename_metadata = parse_filename(os.path.basename(fpath))
                    A = filename_metadata["A"]
                    C = filename_metadata["C"]
                    well_wavelength_ids.append(f"A{A}_C{C}")
                except IndexError:
                    logger.info(f"Skipping {fpath}")
            well_wavelength_ids = sorted(list(set(well_wavelength_ids)))
            actual_wavelength_ids = dict_acquisitions[acquisition][
                "actual_wavelength_ids"
            ]
            if well_wavelength_ids != actual_wavelength_ids:
                raise ValueError(
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
                                        1,
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

            # Prepare AnnData tables for FOV/well ROIs
            well_id = row + column
            FOV_ROIs_table = prepare_FOV_ROI_table(site_metadata.loc[well_id])
            well_ROIs_table = prepare_well_ROI_table(
                site_metadata.loc[well_id]
            )

            # Write AnnData tables into the `tables` zarr group
            write_table(
                group_image,
                "FOV_ROI_table",
                FOV_ROIs_table,
                overwrite=overwrite,
                table_attrs={"type": "roi_table"},
            )
            write_table(
                group_image,
                "well_ROI_table",
                well_ROIs_table,
                overwrite=overwrite,
                table_attrs={"type": "roi_table"},
            )

    # Check that the different images (e.g. different cycles) in the each well
    # have unique labels
    for well_path in zarrurls["well"]:
        check_well_channel_labels(
            well_zarr_path=str(Path(output_path) / well_path)
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
        image_extension=image_extension,
        image_glob_patterns=image_glob_patterns,
    )
    return metadata_update


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=create_ome_zarr_multiplex,
        logger_name=logger.name,
    )

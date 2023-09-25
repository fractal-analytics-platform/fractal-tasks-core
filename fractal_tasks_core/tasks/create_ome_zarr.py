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
Create structure for OME-NGFF zarr array.
"""
import os
from pathlib import Path
from typing import Any
from typing import Optional
from typing import Sequence

import pandas as pd
from pydantic.decorator import validate_arguments

import fractal_tasks_core
from fractal_tasks_core.lib_channels import check_unique_wavelength_ids
from fractal_tasks_core.lib_channels import check_well_channel_labels
from fractal_tasks_core.lib_channels import define_omero_channels
from fractal_tasks_core.lib_channels import OmeroChannel
from fractal_tasks_core.lib_glob import glob_with_multiple_patterns
from fractal_tasks_core.lib_metadata_parsing import parse_yokogawa_metadata
from fractal_tasks_core.lib_parse_filename_metadata import parse_filename
from fractal_tasks_core.lib_regions_of_interest import prepare_FOV_ROI_table
from fractal_tasks_core.lib_regions_of_interest import prepare_well_ROI_table
from fractal_tasks_core.lib_ROI_overlaps import remove_FOV_overlaps
from fractal_tasks_core.lib_write import open_zarr_group_with_overwrite
from fractal_tasks_core.lib_write import write_table


__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__

import logging

logger = logging.getLogger(__name__)


@validate_arguments
def create_ome_zarr(
    *,
    input_paths: Sequence[str],
    output_path: str,
    metadata: dict[str, Any],
    allowed_channels: list[OmeroChannel],
    image_glob_patterns: Optional[list[str]] = None,
    num_levels: int = 5,
    coarsening_xy: int = 2,
    image_extension: str = "tif",
    metadata_table_file: Optional[str] = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """
    Create a OME-NGFF zarr folder, without reading/writing image data.

    Find plates (for each folder in input_paths):

    - glob image files,
    - parse metadata from image filename to identify plates,
    - identify populated channels.

    Create a zarr folder (for each plate):

    - parse mlf metadata,
    - identify wells and field of view (FOV),
    - create FOV ZARR,
    - verify that channels are uniform (i.e., same channels).

    Args:
        input_paths: List of input paths where the image data from
            the microscope is stored (as TIF or PNG).  Should point to the
            parent folder containing the images and the metadata files
            `MeasurementData.mlf` and `MeasurementDetail.mrf` (if present).
            Example: `["/some/path/"]`.
            (standard argument for Fractal tasks, managed by Fractal server).
        output_path: Path were the output of this task is stored.
            Example: "/some/path/" => puts the new OME-Zarr file in the
            "/some/path/".
            (standard argument for Fractal tasks, managed by Fractal server).
        metadata: This parameter is not used by this task.
            (standard argument for Fractal tasks, managed by Fractal server).
        allowed_channels: A list of `OmeroChannel` s, where each channel must
            include the `wavelength_id` attribute and where the
            `wavelength_id` values must be unique across the list.
        image_glob_patterns: If specified, only parse images with filenames
            that match with all these patterns. Patterns must be defined as in
            https://docs.python.org/3/library/fnmatch.html, Example:
            `image_glob_pattern=["*_B03_*"]` => only process well B03
            `image_glob_pattern=["*_C09_*", "*F016*", "*Z[0-5][0-9]C*"]` =>
            only process well C09, field of view 16 and Z planes 0-59.
        num_levels: Number of resolution-pyramid levels. If set to `5`, there
            will be the full-resolution level and 4 levels of
            downsampled images.
        coarsening_xy: Linear coarsening factor between subsequent levels.
            If set to `2`, level 1 is 2x downsampled, level 2 is
            4x downsampled etc.
        image_extension: Filename extension of images (e.g. `"tif"` or `"png"`)
        metadata_table_file: If `None`, parse Yokogawa metadata from mrf/mlf
            files in the input_path folder; else, the full path to a csv file
            containing the parsed metadata table.
        overwrite: If `True`, overwrite the task output.

    Returns:
        A metadata dictionary containing important metadata about the OME-Zarr
            plate, the images and some parameters required by downstream tasks
            (like `num_levels`).
    """

    # Preliminary checks on metadata_table_file
    if metadata_table_file:
        if not metadata_table_file.endswith(".csv"):
            raise ValueError(f"{metadata_table_file=} is not a csv file")
        if not os.path.isfile(metadata_table_file):
            raise FileNotFoundError(f"{metadata_table_file=} does not exist")

    # Identify all plates and all channels, across all input folders
    plates = []
    actual_wavelength_ids = None
    dict_plate_paths = {}
    dict_plate_prefixes: dict[str, Any] = {}

    # Preliminary checks on allowed_channels argument
    check_unique_wavelength_ids(allowed_channels)

    for in_path_str in input_paths:
        in_path = Path(in_path_str)

        # Glob image filenames
        patterns = [f"*.{image_extension}"]
        if image_glob_patterns:
            patterns.extend(image_glob_patterns)
        input_filenames = glob_with_multiple_patterns(
            folder=in_path_str,
            patterns=patterns,
        )

        tmp_wavelength_ids = []
        tmp_plates = []
        for fn in input_filenames:
            try:
                filename_metadata = parse_filename(Path(fn).name)
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
                    f'Skipping "{Path(fn).name}". Original error: ' + str(e)
                )
        tmp_plates = sorted(list(set(tmp_plates)))
        tmp_wavelength_ids = sorted(list(set(tmp_wavelength_ids)))

        info = (
            "Listing plates/channels:\n"
            f"Folder:   {in_path_str}\n"
            f"Patterns: {patterns}\n"
            f"Plates:   {tmp_plates}\n"
            f"Channels: {tmp_wavelength_ids}\n"
        )

        # Check that only one plate is found
        if len(tmp_plates) > 1:
            raise ValueError(f"{info}ERROR: {len(tmp_plates)} plates detected")
        elif len(tmp_plates) == 0:
            raise ValueError(f"{info}ERROR: No plates detected")
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
                raise ValueError(
                    f"ERROR\n{info}\nERROR:"
                    f" expected channels {actual_wavelength_ids}"
                )

        # Update dict_plate_paths
        dict_plate_paths[plate] = in_path

    # Check that all channels are in the allowed_channels
    allowed_wavelength_ids = [
        channel.wavelength_id for channel in allowed_channels
    ]
    if not set(actual_wavelength_ids).issubset(set(allowed_wavelength_ids)):
        msg = "ERROR in create_ome_zarr\n"
        msg += f"actual_wavelength_ids: {actual_wavelength_ids}\n"
        msg += f"allowed_wavelength_ids: {allowed_wavelength_ids}\n"
        raise ValueError(msg)

    # Create actual_channels, i.e. a list of the channel dictionaries which are
    # present
    actual_channels = [
        channel
        for channel in allowed_channels
        if channel.wavelength_id in actual_wavelength_ids
    ]

    zarrurls: dict[str, list[str]] = {"plate": [], "well": [], "image": []}

    ################################################################
    for plate in plates:
        # Define plate zarr
        zarrurl = f"{plate}.zarr"
        in_path = dict_plate_paths[plate]
        logger.info(f"Creating {zarrurl}")
        # Call zarr.open_group wrapper, which handles overwrite=True/False
        group_plate = open_zarr_group_with_overwrite(
            str(Path(output_path) / zarrurl),
            overwrite=overwrite,
        )
        zarrurls["plate"].append(zarrurl)

        # Obtain FOV-metadata dataframe

        if metadata_table_file is None:
            mrf_path = f"{in_path}/MeasurementDetail.mrf"
            mlf_path = f"{in_path}/MeasurementData.mlf"

            site_metadata, number_images_mlf = parse_yokogawa_metadata(
                mrf_path,
                mlf_path,
                filename_patterns=image_glob_patterns,
            )
            site_metadata = remove_FOV_overlaps(site_metadata)

        # If a metadata table was passed, load it and use it directly
        else:
            logger.warning(
                "Since a custom metadata table was provided, there will "
                "be no additional check on the number of image files."
            )
            site_metadata = pd.read_csv(metadata_table_file)
            site_metadata.set_index(["well_id", "FieldIndex"], inplace=True)

        # Extract pixel sizes and bit_depth
        pixel_size_z = site_metadata["pixel_size_z"][0]
        pixel_size_y = site_metadata["pixel_size_y"][0]
        pixel_size_x = site_metadata["pixel_size_x"][0]
        bit_depth = site_metadata["bit_depth"][0]

        if min(pixel_size_z, pixel_size_y, pixel_size_x) < 1e-9:
            raise ValueError(pixel_size_z, pixel_size_y, pixel_size_x)

        # Identify all wells
        plate_prefix = dict_plate_prefixes[plate]

        patterns = [f"{plate_prefix}_*.{image_extension}"]
        if image_glob_patterns:
            patterns.extend(image_glob_patterns)
        plate_images = glob_with_multiple_patterns(
            folder=str(in_path), patterns=patterns
        )

        wells = [
            parse_filename(os.path.basename(fn))["well"] for fn in plate_images
        ]
        wells = sorted(list(set(wells)))

        # Verify that all wells have all channels
        for well in wells:
            patterns = [f"{plate_prefix}_{well}_*.{image_extension}"]
            if image_glob_patterns:
                patterns.extend(image_glob_patterns)
            well_images = glob_with_multiple_patterns(
                folder=str(in_path), patterns=patterns
            )

            # Check number of images matches with expected one
            if metadata_table_file is None:
                num_images_glob = len(well_images)
                num_images_expected = number_images_mlf[well]
                if num_images_glob != num_images_expected:
                    raise ValueError(
                        f"Wrong number of images for {well=}\n"
                        f"Expected {num_images_expected} (from mlf file)\n"
                        f"Found {num_images_glob} files\n"
                        "Other parameters:\n"
                        f"  {image_extension=}\n"
                        f"  {image_glob_patterns=}"
                    )

            well_wavelength_ids = []
            for fpath in well_images:
                try:
                    filename_metadata = parse_filename(os.path.basename(fpath))
                    well_wavelength_ids.append(
                        f"A{filename_metadata['A']}_C{filename_metadata['C']}"
                    )
                except IndexError:
                    logger.info(f"Skipping {fpath}")
            well_wavelength_ids = sorted(list(set(well_wavelength_ids)))
            if well_wavelength_ids != actual_wavelength_ids:
                raise ValueError(
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
                    channels=actual_channels, bit_depth=bit_depth
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
                logger=logger,
            )
            write_table(
                group_image,
                "well_ROI_table",
                well_ROIs_table,
                overwrite=overwrite,
                logger=logger,
            )

    # Check that the different images in each well have unique channel labels.
    # Since we currently merge all fields of view in the same image, this check
    # is useless. It should remain there to catch an error in case we switch
    # back to one-image-per-field-of-view mode
    for well_path in zarrurls["well"]:
        check_well_channel_labels(
            well_zarr_path=str(Path(output_path) / well_path)
        )

    metadata_update = dict(
        plate=zarrurls["plate"],
        well=zarrurls["well"],
        image=zarrurls["image"],
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        image_extension=image_extension,
        image_glob_patterns=image_glob_patterns,
        original_paths=input_paths[:],
    )
    return metadata_update


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=create_ome_zarr,
        logger_name=logger.name,
    )

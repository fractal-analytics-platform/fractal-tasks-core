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

import pandas as pd
from pydantic import validate_call

import fractal_tasks_core
from fractal_tasks_core.cellvoyager.filenames import (
    glob_with_multiple_patterns,
)
from fractal_tasks_core.cellvoyager.filenames import parse_filename
from fractal_tasks_core.cellvoyager.metadata import (
    parse_yokogawa_metadata,
)
from fractal_tasks_core.cellvoyager.metadata import sanitize_string
from fractal_tasks_core.cellvoyager.wells import generate_row_col_split
from fractal_tasks_core.cellvoyager.wells import get_filename_well_id
from fractal_tasks_core.channels import check_unique_wavelength_ids
from fractal_tasks_core.channels import define_omero_channels
from fractal_tasks_core.channels import OmeroChannel
from fractal_tasks_core.ngff.specs import NgffImageMeta
from fractal_tasks_core.ngff.specs import Plate
from fractal_tasks_core.ngff.specs import Well
from fractal_tasks_core.roi import prepare_FOV_ROI_table
from fractal_tasks_core.roi import prepare_well_ROI_table
from fractal_tasks_core.roi import remove_FOV_overlaps
from fractal_tasks_core.tables import write_table
from fractal_tasks_core.tasks.io_models import InitArgsCellVoyager
from fractal_tasks_core.zarr_utils import open_zarr_group_with_overwrite

__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__

import logging

logger = logging.getLogger(__name__)


@validate_call
def cellvoyager_to_ome_zarr_init(
    *,
    # Fractal parameters
    zarr_dir: str,
    # Core parameters
    image_dirs: list[str],
    allowed_channels: list[OmeroChannel],
    # Advanced parameters
    include_glob_patterns: Optional[list[str]] = None,
    exclude_glob_patterns: Optional[list[str]] = None,
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
        zarr_dir: path of the directory where the new OME-Zarrs will be
            created.
            (standard argument for Fractal tasks, managed by Fractal server).
        image_dirs: list of paths to the folders that contains the Cellvoyager
            image files. Each entry is a path to a folder that contains the
            image files themselves for a multiwell plate and the
            MeasurementData & MeasurementDetail metadata files.
        allowed_channels: A list of `OmeroChannel` s, where each channel must
            include the `wavelength_id` attribute and where the
            `wavelength_id` values must be unique across the list.
        include_glob_patterns: If specified, only parse images with filenames
            that match with all these patterns. Patterns must be defined as in
            https://docs.python.org/3/library/fnmatch.html, Example:
            `image_glob_pattern=["*_B03_*"]` => only process well B03
            `image_glob_pattern=["*_C09_*", "*F016*", "*Z[0-5][0-9]C*"]` =>
            only process well C09, field of view 16 and Z planes 0-59.
            Can interact with exclude_glob_patterns: All included images - all
            excluded images gives the final list of images to process
        exclude_glob_patterns: If specified, exclude any image where the
            filename matches any of the exclusion patterns. Patterns are
            specified the same as for include_glob_patterns.
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

    for image_dir in image_dirs:
        # Glob image filenames
        include_patterns = [f"*.{image_extension}"]
        exclude_patterns = []
        if include_glob_patterns:
            include_patterns.extend(include_glob_patterns)
        if exclude_glob_patterns:
            exclude_patterns.extend(exclude_glob_patterns)
        input_filenames = glob_with_multiple_patterns(
            folder=image_dir,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
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
            f"Folder:   {image_dir}\n"
            f"Include Patterns: {include_patterns}\n"
            f"Exclude Patterns: {exclude_patterns}\n"
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
        dict_plate_paths[plate] = image_dir

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

    ################################################################
    # Create well/image OME-Zarr folders on disk, and prepare output
    # metadata
    parallelization_list = []

    for plate in plates:
        plate_name = sanitize_string(plate)
        # Define plate zarr
        relative_zarrurl = f"{plate_name}.zarr"
        in_path = dict_plate_paths[plate]
        logger.info(f"Creating {relative_zarrurl}")
        # Call zarr.open_group wrapper, which handles overwrite=True/False
        group_plate = open_zarr_group_with_overwrite(
            str(Path(zarr_dir) / relative_zarrurl),
            overwrite=overwrite,
        )

        # Obtain FOV-metadata dataframe
        if metadata_table_file is None:
            mrf_path = f"{in_path}/MeasurementDetail.mrf"
            mlf_path = f"{in_path}/MeasurementData.mlf"

            site_metadata, number_images_mlf = parse_yokogawa_metadata(
                mrf_path,
                mlf_path,
                include_patterns=include_glob_patterns,
                exclude_patterns=exclude_glob_patterns,
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
        pixel_size_z = site_metadata["pixel_size_z"].iloc[0]
        pixel_size_y = site_metadata["pixel_size_y"].iloc[0]
        pixel_size_x = site_metadata["pixel_size_x"].iloc[0]
        bit_depth = site_metadata["bit_depth"].iloc[0]

        if min(pixel_size_z, pixel_size_y, pixel_size_x) < 1e-9:
            raise ValueError(pixel_size_z, pixel_size_y, pixel_size_x)

        # Identify all wells
        plate_prefix = dict_plate_prefixes[plate]

        include_patterns = [f"{plate_prefix}_*.{image_extension}"]
        if include_glob_patterns:
            include_patterns.extend(include_glob_patterns)
        plate_images = glob_with_multiple_patterns(
            folder=str(in_path),
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )

        wells = [
            parse_filename(os.path.basename(fn))["well"] for fn in plate_images
        ]
        wells = sorted(list(set(wells)))

        # Verify that all wells have all channels
        for well in wells:
            include_patterns = [f"{plate_prefix}_{well}_*.{image_extension}"]
            if include_glob_patterns:
                include_patterns.extend(include_glob_patterns)
            well_images = glob_with_multiple_patterns(
                folder=str(in_path),
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
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
                        f"  {include_glob_patterns=}\n"
                        f"  {exclude_glob_patterns=}\n"
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
                    f"ERROR: well {well} in plate {plate_name} (prefix: "
                    f"{plate_prefix}) has missing channels.\n"
                    f"Expected: {actual_channels}\n"
                    f"Found: {well_wavelength_ids}.\n"
                )

        well_rows_columns = generate_row_col_split(wells)

        row_list = [
            well_row_column[0] for well_row_column in well_rows_columns
        ]
        col_list = [
            well_row_column[1] for well_row_column in well_rows_columns
        ]
        row_list = sorted(list(set(row_list)))
        col_list = sorted(list(set(col_list)))

        plate_attrs = {
            "acquisitions": [{"id": 0, "name": plate_name}],
            "columns": [{"name": col} for col in col_list],
            "rows": [{"name": row} for row in row_list],
            "version": __OME_NGFF_VERSION__,
            "wells": [
                {
                    "path": well_row_column[0] + "/" + well_row_column[1],
                    "rowIndex": row_list.index(well_row_column[0]),
                    "columnIndex": col_list.index(well_row_column[1]),
                }
                for well_row_column in well_rows_columns
            ],
        }

        # Validate plate attrs:
        Plate(**plate_attrs)

        group_plate.attrs["plate"] = plate_attrs

        for row, column in well_rows_columns:
            parallelization_list.append(
                {
                    "zarr_url": (
                        f"{zarr_dir}/{plate_name}.zarr/{row}/{column}/0"
                    ),
                    "init_args": InitArgsCellVoyager(
                        image_dir=in_path,
                        plate_prefix=plate_prefix,
                        well_ID=get_filename_well_id(row, column),
                        image_extension=image_extension,
                        include_glob_patterns=include_glob_patterns,
                        exclude_glob_patterns=exclude_glob_patterns,
                    ).model_dump(),
                }
            )
            group_well = group_plate.create_group(f"{row}/{column}/")

            well_attrs = {
                "images": [{"path": "0"}],
                "version": __OME_NGFF_VERSION__,
            }

            # Validate well attrs:
            Well(**well_attrs)
            group_well.attrs["well"] = well_attrs

            group_image = group_well.create_group("0")  # noqa: F841
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
                "id": 1,  # TODO does this depend on the plate number?
                "name": "TBD",
                "version": __OME_NGFF_VERSION__,
                "channels": define_omero_channels(
                    channels=actual_channels, bit_depth=bit_depth
                ),
            }

            # Validate Image attrs
            NgffImageMeta(**group_image.attrs)

            # Prepare AnnData tables for FOV/well ROIs
            well_id = get_filename_well_id(row, column)
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

    return dict(parallelization_list=parallelization_list)


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=cellvoyager_to_ome_zarr_init,
        logger_name=logger.name,
    )

# Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Joel Lüthi  <joel.luethi@fmi.ch>
# Tommaso Comparin <tommaso.comparin@exact-lab.it>
#
# This file is part of Fractal and was originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
"""
Functions to create a metadata dataframe from Yokogawa files.
"""
import fnmatch
import logging
import math
from pathlib import Path
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
from defusedxml import ElementTree

logger = logging.getLogger(__name__)


def parse_yokogawa_metadata(
    mrf_path: Union[str, Path],
    mlf_path: Union[str, Path],
    *,
    filename_patterns: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Parse Yokogawa CV7000 metadata files and prepare site-level metadata.

    Args:
        mrf_path: Full path to MeasurementDetail.mrf metadata file.
        mlf_path: Full path to MeasurementData.mlf metadata file.
        filename_patterns:
            List of patterns to filter the image filenames in the mlf metadata
            table. Patterns must be defined as in
            https://docs.python.org/3/library/fnmatch.html
    """

    # Convert paths to strings
    mrf_str = Path(mrf_path).as_posix()
    mlf_str = Path(mlf_path).as_posix()

    mrf_frame, mlf_frame, error_count = read_metadata_files(
        mrf_str, mlf_str, filename_patterns
    )

    # Aggregate information from the mlf file
    per_site_parameters = ["X", "Y"]

    grouping_params = ["well_id", "FieldIndex"]
    grouped_sites = mlf_frame.loc[
        :, grouping_params + per_site_parameters
    ].groupby(by=grouping_params)

    check_group_consistency(grouped_sites, message="X & Y stage positions")
    site_metadata = grouped_sites.mean()
    site_metadata.columns = ["x_micrometer", "y_micrometer"]
    site_metadata["z_micrometer"] = 0

    site_metadata = pd.concat(
        [
            site_metadata,
            get_z_steps(mlf_frame),
            get_earliest_time_per_site(mlf_frame),
        ],
        axis=1,
    )

    # Aggregate information from the mrf file
    mrf_columns = [
        "horiz_pixel_dim",
        "vert_pixel_dim",
        "horiz_pixels",
        "vert_pixels",
        "bit_depth",
    ]
    check_group_consistency(
        mrf_frame.loc[:, mrf_columns], message="Image dimensions"
    )
    site_metadata["pixel_size_x"] = mrf_frame.loc[:, "horiz_pixel_dim"].max()
    site_metadata["pixel_size_y"] = mrf_frame.loc[:, "vert_pixel_dim"].max()
    site_metadata["x_pixel"] = int(mrf_frame.loc[:, "horiz_pixels"].max())
    site_metadata["y_pixel"] = int(mrf_frame.loc[:, "vert_pixels"].max())
    site_metadata["bit_depth"] = int(mrf_frame.loc[:, "bit_depth"].max())

    if error_count > 0:
        logger.info(
            f"There were {error_count} ERR entries in the metadatafile. "
            f"Still succesfully parsed {len(site_metadata)} sites. "
        )

    # Compute expected number of image files for each well
    list_of_wells = set(site_metadata.index.get_level_values("well_id"))
    number_of_files = {}
    for this_well_id in list_of_wells:
        num_images = (mlf_frame.well_id == this_well_id).sum()
        logger.info(
            f"Expected number of images for well {this_well_id}: {num_images}"
        )
        number_of_files[this_well_id] = num_images
    # Check that the sum of per-well file numbers correspond to the total
    # file number
    if not sum(number_of_files.values()) == len(mlf_frame):
        raise ValueError(
            "Error while counting the number of image files per well.\n"
            f"{len(mlf_frame)=}\n"
            f"{number_of_files=}"
        )

    return site_metadata, number_of_files


def read_metadata_files(
    mrf_path: str,
    mlf_path: str,
    filename_patterns: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Create tables for mrf & mlf Yokogawa metadata.

    Args:
        mrf_path: Full path to MeasurementDetail.mrf metadata file.
        mlf_path: Full path to MeasurementData.mlf metadata file.
        filename_patterns: List of patterns to filter the image filenames in
            the mlf metadata table. Patterns must be defined as in
            https://docs.python.org/3/library/fnmatch.html.

    Returns:

    """

    # parsing of mrf & mlf files are based on the
    # yokogawa_image_collection_task v0.5 in drogon, written by Dario Vischi.
    # https://github.com/fmi-basel/job-system-workflows/blob/00bbf34448972d27f258a2c28245dd96180e8229/src/gliberal_workflows/tasks/yokogawa_image_collection_task/versions/version_0_5.py  # noqa
    # Now modified for Fractal use

    mrf_frame, plate_type = read_mrf_file(mrf_path)

    # filter_position & filter_wheel_position are parsed, but not
    # processed further. Figure out how to save them as relevant metadata for
    # use e.g. during illumination correction

    mlf_frame, error_count = read_mlf_file(
        mlf_path, plate_type, filename_patterns
    )
    # Time points are parsed as part of the mlf_frame, but currently not
    # processed further. Once we tackle time-resolved data, parse from here.

    return mrf_frame, mlf_frame, error_count


def read_mrf_file(mrf_path: str) -> tuple[pd.DataFrame, int]:
    """
    Parses the mrf metadata file

    Args:
        mrf_path: Full path to MeasurementDetail.mrf metadata file.

    Returns:
        Parsed mrf pandas table with one row per channel imaged
        The plate_type: The number of wells
    """

    # Prepare mrf dataframe
    mrf_columns = [
        "channel_id",
        "horiz_pixel_dim",
        "vert_pixel_dim",
        "camera_no",
        "bit_depth",
        "horiz_pixels",
        "vert_pixels",
        "filter_wheel_position",
        "filter_position",
        "shading_corr_src",
    ]
    mrf_frame = pd.DataFrame(columns=mrf_columns)

    mrf_xml = ElementTree.parse(mrf_path).getroot()

    # Read mrf file
    ns = {"bts": "http://www.yokogawa.co.jp/BTS/BTSSchema/1.0"}
    # Fetch RowCount and ColumnCount
    row_count = int(mrf_xml.attrib[f'{{{ns["bts"]}}}RowCount'])
    column_count = int(mrf_xml.attrib[f'{{{ns["bts"]}}}ColumnCount'])
    plate_type = row_count * column_count

    for channel in mrf_xml.findall("bts:MeasurementChannel", namespaces=ns):
        mrf_frame.loc[channel.get("{%s}Ch" % ns["bts"])] = [
            channel.get("{%s}Ch" % ns["bts"]),
            float(channel.get("{%s}HorizontalPixelDimension" % ns["bts"])),
            float(channel.get("{%s}VerticalPixelDimension" % ns["bts"])),
            int(channel.get("{%s}CameraNumber" % ns["bts"])),
            int(channel.get("{%s}InputBitDepth" % ns["bts"])),
            int(channel.get("{%s}HorizontalPixels" % ns["bts"])),
            int(channel.get("{%s}VerticalPixels" % ns["bts"])),
            int(channel.get("{%s}FilterWheelPosition" % ns["bts"])),
            int(channel.get("{%s}FilterPosition" % ns["bts"])),
            channel.get("{%s}ShadingCorrectionSource" % ns["bts"]),
        ]

    return mrf_frame, plate_type


def _create_well_ids(
    row_series: pd.Series,
    col_series: pd.Series,
    plate_type: int,
) -> list[str]:
    """
    Create well_id list from XML metadata

    Handles the conversion of Cellvoyager XML metadata into well indentifiers.
    Returns well identifiers like A01, B02 etc. for 96 & 384 well plates.
    Returns well identifiers like A01.a1, A01.b2 etc. for 1536 well plates.
    Defaults to the processing used for 96 & 384 well plates, unless the
    plate_type is 1536. For 1536 well plates, the first 4x4 wells go into
    A01.a1 - A01.d4 and so on.

    Args:
        row_series: Series with index being the index of the image and the
            value the row position (starting at 1 for top left).
        col_series: Series with index being the index of the image and the
            value the col position (starting at 1 for top left).
        plate_type: Number of wells in the plate layout. Used to determine
            whether it's a 1536 well plate or a different layout.

    Returns:
        list of well_ids

    """
    if plate_type == 1536:
        # Row are built of a base letter (matching to the 96 well plate layout)
        # and a sub letter (position of the 1536 well within the 4x4 grid,
        # can be a-d) of that well
        row_base = [chr(math.floor((x - 1) / 4) + 65) for x in (row_series)]
        row_sub = [chr((x - 1) % 4 + 97) for x in (row_series)]
        # Columns are built of a base number (matching to the 96 well plate
        # layout) and a sub integer (position of the 1536 well within the
        # 4x4 grid, can be 1-4) of that well
        col_base = [math.floor((x - 1) / 4) + 1 for x in col_series]
        col_sub = [(x - 1) % 4 + 1 for x in col_series]
        well_ids = []
        for i in range(len(row_base)):
            well_ids.append(
                f"{row_base[i]}{col_base[i]:02}.{row_sub[i]}{col_sub[i]}"
            )
    else:
        row_str = [chr(x) for x in (row_series + 64)]
        well_ids = [f"{a}{b:02}" for a, b in zip(row_str, col_series)]

    return well_ids


def read_mlf_file(
    mlf_path: str,
    plate_type: int,
    filename_patterns: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, int]:
    """
    Process the mlf metadata file of a Cellvoyager CV7K/CV8K.

    Args:
        mlf_path: Full path to MeasurementData.mlf metadata file.
        plate_type: Plate layout, integer for the number of potential wells.
        filename_patterns: List of patterns to filter the image filenames in
            the mlf metadata table. Patterns must be defined as in
            https://docs.python.org/3/library/fnmatch.html.

    Returns:
        mlf_frame: pd.DataFrame with relevant metadata per image
        error_count: Count of errors found during metadata processing
    """

    # Load the whole MeasurementData.mlf file
    mlf_frame_raw = pd.read_xml(mlf_path)

    # Remove all rows that do not match the given patterns
    logger.info(
        f"Read {mlf_path}, and apply following patterns to "
        f"image filenames: {filename_patterns}"
    )
    if filename_patterns:
        filenames = mlf_frame_raw.MeasurementRecord
        keep_row = None
        for pattern in filename_patterns:
            actual_pattern = fnmatch.translate(pattern)
            new_matches = filenames.str.fullmatch(actual_pattern)
            if new_matches.sum() == 0:
                raise ValueError(
                    f"In {mlf_path} there is no image filename "
                    f'matching "{actual_pattern}".'
                )
            if keep_row is None:
                keep_row = new_matches.copy()
            else:
                keep_row = keep_row & new_matches
        if keep_row.sum() == 0:
            raise ValueError(
                f"In {mlf_path} there is no image filename "
                f"matching {filename_patterns}."
            )
        mlf_frame_matching = mlf_frame_raw[keep_row.values].copy()
    else:
        mlf_frame_matching = mlf_frame_raw.copy()

    # Create a well ID column
    # Row & column are provided as int from XML metadata
    mlf_frame_matching["well_id"] = _create_well_ids(
        mlf_frame_matching["Row"], mlf_frame_matching["Column"], plate_type
    )

    # Flip Y axis to align to image coordinate system
    mlf_frame_matching["Y"] = -mlf_frame_matching["Y"]

    # Compute number or errors
    error_count = (mlf_frame_matching["Type"] == "ERR").sum()

    # We're only interested in the image metadata
    mlf_frame = mlf_frame_matching[mlf_frame_matching["Type"] == "IMG"]

    return mlf_frame, error_count


def calculate_steps(site_series: pd.Series):
    """
    TBD

    Args:
        site_series: TBD
    """

    # site_series is the z_micrometer series for a given site of a given
    # channel. This function calculates the step size in Z

    # First diff is always NaN because there is nothing to compare it to
    steps = site_series.diff().dropna().astype(float)
    if not np.allclose(steps.iloc[0], np.array(steps)):
        raise NotImplementedError(
            "When parsing the Yokogawa mlf file, some sites "
            "had varying step size in Z. "
            "That is not supported for the OME-Zarr parsing"
        )
    return steps.mean()


def get_z_steps(mlf_frame: pd.DataFrame) -> pd.DataFrame:
    """
    TBD

    Args:
        mlf_frame: TBD
    """

    # Process mlf_frame to extract Z information (pixel size & steps).
    # Run checks on consistencies & return site-based z step dataframe
    # Group by well, field & channel
    grouped_sites_z = (
        mlf_frame.loc[
            :,
            ["well_id", "FieldIndex", "ActionIndex", "Ch", "Z"],
        ]
        .set_index(["well_id", "FieldIndex", "ActionIndex", "Ch"])
        .groupby(level=[0, 1, 2, 3])
    )

    # If there is only 1 Z step, set the Z spacing to the count of planes => 1
    if grouped_sites_z.count()["Z"].max() == 1:
        z_data = grouped_sites_z.count().groupby(["well_id", "FieldIndex"])
    else:
        # Group the whole site (combine channels), because Z steps need to be
        # consistent between channels for OME-Zarr.
        z_data = grouped_sites_z.apply(calculate_steps).groupby(
            ["well_id", "FieldIndex"]
        )

    check_group_consistency(
        z_data, message="Comparing Z steps between channels"
    )

    # Ensure that channels have the same number of z planes and
    # reduce it to one value.
    # Only check if there is more than one channel available
    if any(
        grouped_sites_z.count().groupby(["well_id", "FieldIndex"]).count() > 1
    ):
        check_group_consistency(
            grouped_sites_z.count().groupby(["well_id", "FieldIndex"]),
            message="Checking number of Z steps between channels",
        )

    z_steps = (
        grouped_sites_z.count()
        .groupby(["well_id", "FieldIndex"])
        .mean()
        .astype(int)
    )

    # Combine the two dataframes
    z_frame = pd.concat([z_data.mean(), z_steps], axis=1)
    z_frame.columns = ["pixel_size_z", "z_pixel"]
    return z_frame


def get_earliest_time_per_site(mlf_frame: pd.DataFrame) -> pd.DataFrame:
    """
    TBD

    Args:
        mlf_frame: TBD
    """

    # Get the time information per site
    # Because a site will contain time information for each plane
    # of each channel, we just return the earliest time infromation
    # per site.
    return pd.to_datetime(
        mlf_frame.groupby(["well_id", "FieldIndex"]).min()["Time"], utc=True
    )


def check_group_consistency(grouped_df: pd.DataFrame, message: str = ""):
    """
    TBD

    Args:
        grouped_df: TBD
        message: TBD
    """

    # Check consistency in grouped df for multi-index, multi-column dataframes
    # raises an exception if there is variability
    diff_df = grouped_df.max() - grouped_df.min()
    if not np.isclose(np.sum(np.sum(diff_df)), 0.0):
        raise ValueError(
            "During metadata parsing, a consistency check failed: \n"
            f"{message}\n"
            f"Difference dataframe: \n{diff_df}"
        )

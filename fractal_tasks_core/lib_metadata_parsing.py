"""
Copyright 2022 (C)
    Friedrich Miescher Institute for Biomedical Research and
    University of Zurich

    Original authors:
    Joel Lüthi  <joel.luethi@fmi.ch>

    This file is part of Fractal and was originally developed by eXact lab
    S.r.l.  <exact-lab.it> under contract with Liberali Lab from the Friedrich
    Miescher Institute for Biomedical Research and Pelkmans Lab from the
    University of Zurich.

Functions to create a metadata dataframe from Yokogawa files
"""
import logging

import numpy as np
import pandas as pd
from defusedxml import ElementTree

logger = logging.getLogger(__name__)


def parse_yokogawa_metadata(mrf_path, mlf_path):
    """
    Parse Yokogawa CV7000 metadata files and prepare site-level metadata

    :param mrf_path: Full path to MeasurementDetail.mrf metadata file
    :type mrf_path: Union(str, pathlib.Path)
    :param mlf_path: Full path to MeasurementData.mlf metadata file
    :type mlf_path: Union(str, pathlib.Path)
    """
    mrf_frame, mlf_frame, error_count = read_metadata_files(mrf_path, mlf_path)

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
    total_files = len(mlf_frame)
    # TODO: Check whether the total_files correspond to the number of
    # relevant input images in the input folder. Returning it for now
    # Maybe return it here for further checks and produce a warning if it does
    # not match

    return site_metadata, total_files


def read_metadata_files(mrf_path, mlf_path):
    """
    Description

    :param dummy: this is just a placeholder
    :type dummy: int
    """

    # parsing of mrf & mlf files are based on the
    # yokogawa_image_collection_task v0.5 in drogon, written by Dario Vischi.
    # https://github.com/fmi-basel/job-system-workflows/blob/00bbf34448972d27f258a2c28245dd96180e8229/src/gliberal_workflows/tasks/yokogawa_image_collection_task/versions/version_0_5.py  # noqa
    # Now modified for Fractal use

    mrf_frame = read_mrf_file(mrf_path)
    # TODO: filter_position & filter_wheel_position are parsed, but not
    # processed further. Figure out how to save them as relevant metadata for
    # use e.g. during illumination correction

    mlf_frame, error_count = read_mlf_file(mlf_path)
    # TODO: Time points are parsed as part of the mlf_frame, but currently not
    # processed further. Once we tackle time-resolved data, parse from here.

    return mrf_frame, mlf_frame, error_count


def read_mrf_file(mrf_path):
    """
    Description

    :param dummy: this is just a placeholder
    :type dummy: int
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

    return mrf_frame


def read_mlf_file(mlf_path):
    """
    Description

    :param dummy: this is just a placeholder
    :type dummy: int
    """

    mlf_frame_raw = pd.read_xml(mlf_path)

    # Create a well ID column
    row_str = [chr(x) for x in (mlf_frame_raw["Row"] + 64)]
    mlf_frame_raw["well_id"] = [
        "{}{:02}".format(a, b)
        for a, b in zip(row_str, mlf_frame_raw["Column"])
    ]

    # Flip Y axis to align to image coordinate system
    mlf_frame_raw["Y"] = -mlf_frame_raw["Y"]

    # We're only interested in the image metadata
    mlf_frame = mlf_frame_raw[mlf_frame_raw["Type"] == "IMG"]

    error_count = (mlf_frame_raw["Type"] == "ERR").sum()

    return mlf_frame, error_count


def calculate_steps(site_series: pd.Series):
    """
    Description

    :param dummy: this is just a placeholder
    :type dummy: int
    """

    # site_series is the z_micrometer series for a given site of a given
    # channel. This function calculates the step size in Z

    # First diff is always NaN because there is nothing to compare it to
    steps = site_series.diff().dropna().astype(float)
    if not np.allclose(steps.iloc[0], np.array(steps)):
        raise Exception(
            "When parsing the Yokogawa mlf file, some sites "
            "had varying step size in Z. "
            "That is not supported for the OME-Zarr parsing"
        )
    return steps.mean()


def get_z_steps(mlf_frame):
    """
    Description

    :param dummy: this is just a placeholder
    :type dummy: int
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


def get_earliest_time_per_site(mlf_frame) -> pd.DataFrame:
    """
    Description

    :param dummy: this is just a placeholder
    :type dummy: int
    """

    # Get the time information per site
    # Because a site will contain time information for each plane
    # of each channel, we just return the earliest time infromation
    # per site.
    return pd.to_datetime(
        mlf_frame.groupby(["well_id", "FieldIndex"]).min()["Time"], utc=True
    )


def check_group_consistency(grouped_df, message=""):
    """
    Description

    :param dummy: this is just a placeholder
    :type dummy: int
    """

    # Check consistency in grouped df for multi-index, multi-column dataframes
    # raises an exception if there is variability
    diff_df = grouped_df.max() - grouped_df.min()
    if not np.isclose(np.sum(np.sum(diff_df)), 0.0):
        raise Exception(
            "During metadata parsing, a consistency check failed: \n"
            f"{message}\n"
            f"Difference dataframe: \n{diff_df}"
        )

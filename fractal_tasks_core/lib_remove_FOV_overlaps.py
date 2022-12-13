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

Functions to identify and remove overlaps between regions of interest
"""
import logging
from typing import Sequence
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def is_overlapping_1D(
    line1: Sequence[float], line2: Sequence[float], tol: float = 0
) -> bool:
    """
    Given two intervals, finds whether they overlap

    This is based on https://stackoverflow.com/a/70023212/19085332, and we
    additionally use a finite tolerance for floating-point comparisons.

    :param line1: The boundaries of the first interval , written as
                  ``[x_min, x_max]``.
    :param line2: The boundaries of the second interval , written as
                  ``[x_min, x_max]``.
    :param tol: Finite tolerance for floating-point comparisons.
    """
    return line1[0] <= line2[1] - tol and line2[0] <= line1[1] - tol


def is_overlapping_2D(
    box1: Sequence[float], box2: Sequence[float], tol: float = 0
) -> bool:
    """
    Given two rectangular boxes, finds whether they overlap

    This is based on https://stackoverflow.com/a/70023212/19085332, and we
    additionally use a finite tolerance for floating-point comparisons.

    :param box1: The boundaries of the first rectangle, written as
                 ``[x_min, y_min, x_max, y_max]``.
    :param box2: The boundaries of the second rectangle, written as
                 ``[x_min, y_min, x_max, y_max]``.
    :param tol: Finite tolerance for floating-point comparisons.
    """
    overlap_x = is_overlapping_1D(
        [box1[0], box1[2]], [box2[0], box2[2]], tol=tol
    )
    overlap_y = is_overlapping_1D(
        [box1[1], box1[3]], [box2[1], box2[3]], tol=tol
    )
    return overlap_x and overlap_y


def is_overlapping_3D(box1, box2, tol=0):
    """
    Given two three-dimensional boxes, finds whether they overlap

    This is based on https://stackoverflow.com/a/70023212/19085332, and we
    additionally use a finite tolerance for floating-point comparisons.

    :param box1: The boundaries of the first box, written as
                 ``[x_min, y_min, z_min, x_max, y_max, z_max]``.
    :param box2: The boundaries of the second box, written as
                 ``[x_min, y_min, z_min, x_max, y_max, z_max]``.
    :param tol: Finite tolerance for floating-point comparisons.
    """

    overlap_x = is_overlapping_1D(
        [box1[0], box1[3]], [box2[0], box2[3]], tol=tol
    )
    overlap_y = is_overlapping_1D(
        [box1[1], box1[4]], [box2[1], box2[4]], tol=tol
    )
    overlap_z = is_overlapping_1D(
        [box1[2], box1[5]], [box2[2], box2[5]], tol=tol
    )
    return overlap_x and overlap_y and overlap_z


def get_overlapping_pair(tmp_df: pd.DataFrame, tol: float = 0) -> Tuple[int]:
    """
    Finds the indices for the next overlapping FOVs pair

    Note: the returned indices are positional indices, starting from 0

    :param tmp_df: Dataframe with columns `["xmin", "ymin", "xmax", "ymax"]`.
    :param tol: Finite tolerance for floating-point comparisons.
    """

    num_lines = len(tmp_df.index)
    for pos_ind_1 in range(num_lines):
        for pos_ind_2 in range(pos_ind_1):
            if is_overlapping_2D(
                tmp_df.iloc[pos_ind_1], tmp_df.iloc[pos_ind_2], tol=tol
            ):
                return (pos_ind_1, pos_ind_2)
    return False


def get_overlapping_pairs_3D(tmp_df: pd.DataFrame, pixel_sizes):
    """
    Finds the indices for the next overlapping FOVs pair, in three dimensions

    Note: the returned indices are positional indices, starting from 0

    :param tmp_df: Dataframe with columns ``{x,y,z}_micrometer`` and
                   ``len_{x,y,z}_micrometer``.
    :param tol: Finite tolerance for floating-point comparisons.
    """
    tol = 1e-10
    if tol > min(pixel_sizes) / 1e3:
        raise Exception(f"{tol=} but {pixel_sizes=}")
    new_tmp_df = tmp_df.copy()

    new_tmp_df["x_micrometer_max"] = (
        new_tmp_df["x_micrometer"] + new_tmp_df["len_x_micrometer"]
    )
    new_tmp_df["y_micrometer_max"] = (
        new_tmp_df["y_micrometer"] + new_tmp_df["len_y_micrometer"]
    )
    new_tmp_df["z_micrometer_max"] = (
        new_tmp_df["z_micrometer"] + new_tmp_df["len_z_micrometer"]
    )
    new_tmp_df.drop(labels=["len_x_micrometer"], axis=1, inplace=True)
    new_tmp_df.drop(labels=["len_y_micrometer"], axis=1, inplace=True)
    new_tmp_df.drop(labels=["len_z_micrometer"], axis=1, inplace=True)
    num_lines = len(new_tmp_df.index)
    overlapping_list = []
    # pos_ind_1 and pos_ind_2 are labels value
    for pos_ind_1 in range(num_lines):
        for pos_ind_2 in range(pos_ind_1):
            if is_overlapping_3D(
                new_tmp_df.iloc[pos_ind_1], new_tmp_df.iloc[pos_ind_2], tol=tol
            ):
                # we accumulate tuples of overlapping labels
                overlapping_list.append((pos_ind_1, pos_ind_2))
    if len(overlapping_list) > 0:
        raise ValueError(
            f"{overlapping_list} " f"List of pair of bounding box overlaps"
        )
    return overlapping_list


def apply_shift_in_one_direction(
    tmp_df_well: pd.DataFrame,
    line_1: Sequence[float],
    line_2: Sequence[float],
    mu: str,
    tol: float = 0,
):
    min_1, max_1 = line_1[:]
    min_2, max_2 = line_2[:]
    min_max = min(max_1, max_2)
    max_min = max(min_1, min_2)
    shift = min_max - max_min
    logging.debug(f"{mu}-shifting by {shift=}")
    ind = tmp_df_well.loc[:, f"{mu}min"] >= max_min - tol
    if not (shift > 0.0 and ind.to_numpy().max() > 0):
        raise ValueError(
            "Something wrong in apply_shift_in_one_direction\n"
            f"{mu=}\n{shift=}\n{ind.to_numpy()=}"
        )
    tmp_df_well.loc[ind, f"{mu}min"] += shift
    tmp_df_well.loc[ind, f"{mu}max"] += shift
    tmp_df_well.loc[ind, f"{mu}_micrometer"] += shift
    return tmp_df_well


def remove_FOV_overlaps(df: pd.DataFrame):
    """
    Given a metadata dataframe, shift its columns to remove FOV overlaps

    :param df: Metadata dataframe
    """

    # Set tolerance (this should be much smaller than pixel size or expected
    # round-offs), and maximum number of iterations in constraint solver
    tol = 1e-10
    max_iterations = 200

    # Create a local copy of the dataframe
    df = df.copy()

    # Create temporary columns (to streamline overlap removals), which are
    # then removed at the end of the remove_FOV_overlaps function
    df["xmin"] = df["x_micrometer"]
    df["ymin"] = df["y_micrometer"]
    df["xmax"] = df["x_micrometer"] + df["pixel_size_x"] * df["x_pixel"]
    df["ymax"] = df["y_micrometer"] + df["pixel_size_y"] * df["y_pixel"]
    list_columns = ["xmin", "ymin", "xmax", "ymax"]

    # Create columns with the original positions (not to be removed)
    df["x_micrometer_original"] = df["x_micrometer"]
    df["y_micrometer_original"] = df["y_micrometer"]

    # Check that tolerance is much smaller than pixel sizes
    min_pixel_size = df[["pixel_size_x", "pixel_size_y"]].min().min()
    if tol > min_pixel_size / 1e3:
        raise Exception(
            f"In remove_FOV_overlaps, {tol=} but {min_pixel_size=}"
        )

    # Loop over wells
    wells = sorted(list(set([ind[0] for ind in df.index])))
    for well in wells:

        logger.info(f"removing FOV overlaps for {well=}")
        df_well = df.loc[well].copy()

        # NOTE: these are positional indices (i.e. starting from 0)
        pair_pos_indices = get_overlapping_pair(df_well[list_columns], tol=tol)

        # Keep going until there are no overlaps, or until iteration reaches
        # max_iterations
        iteration = 0
        while pair_pos_indices:
            iteration += 1

            # Identify overlapping FOVs
            pos_ind_1, pos_ind_2 = pair_pos_indices
            fov_id_1 = df_well.index[pos_ind_1]
            fov_id_2 = df_well.index[pos_ind_2]
            xmin_1, ymin_1, xmax_1, ymax_1 = df_well[list_columns].iloc[
                pos_ind_1
            ]
            xmin_2, ymin_2, xmax_2, ymax_2 = df_well[list_columns].iloc[
                pos_ind_2
            ]
            logger.debug(
                f"{well=}, {iteration=}, removing overlap between"
                f" {fov_id_1=} and {fov_id_2=}"
            )

            # Check what kind of overlap is there (X, Y, or XY)
            is_x_equal = abs(xmin_1 - xmin_2) < tol and (xmax_1 - xmax_2) < tol
            is_y_equal = abs(ymin_1 - ymin_2) < tol and (ymax_1 - ymax_2) < tol
            is_x_overlap = is_overlapping_1D(
                [xmin_1, xmax_1], [xmin_2, xmax_2], tol=tol
            )
            is_y_overlap = is_overlapping_1D(
                [ymin_1, ymax_1], [ymin_2, ymax_2], tol=tol
            )

            if is_x_equal and is_y_overlap:
                # Y overlap
                df_well = apply_shift_in_one_direction(
                    df_well,
                    [ymin_1, ymax_1],
                    [ymin_2, ymax_2],
                    mu="y",
                    tol=tol,
                )
            elif is_y_equal and is_x_overlap:
                # X overlap
                df_well = apply_shift_in_one_direction(
                    df_well,
                    [xmin_1, xmax_1],
                    [xmin_2, xmax_2],
                    mu="x",
                    tol=tol,
                )
            elif not (is_x_equal or is_y_equal) and (
                is_x_overlap and is_y_overlap
            ):
                # XY overlap
                df_well = apply_shift_in_one_direction(
                    df_well,
                    [xmin_1, xmax_1],
                    [xmin_2, xmax_2],
                    mu="x",
                    tol=tol,
                )
                df_well = apply_shift_in_one_direction(
                    df_well,
                    [ymin_1, ymax_1],
                    [ymin_2, ymax_2],
                    mu="y",
                    tol=tol,
                )
            else:
                raise ValueError(
                    "Trying to remove overlap which is not there."
                )

            # Look for next overlapping FOV pair
            pair_pos_indices = get_overlapping_pair(
                df_well[list_columns], tol=tol
            )

            # Enforce maximum number of iterations
            if iteration >= max_iterations:
                raise ValueError(f"Reached {max_iterations=} for {well=}")

        # Note: using df.loc[well] = df_well leads to a NaN dataframe, see
        # for instance https://stackoverflow.com/a/28432733/19085332
        df.loc[well, :] = df_well.values

    # Remove temporary columns that were added only as part of this function
    df.drop(list_columns, axis=1, inplace=True)

    return df

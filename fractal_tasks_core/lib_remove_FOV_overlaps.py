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

logger = logging.getLogger(__name__)


def is_overlapping_1D(line1, line2, tol=0):
    """
    Based on https://stackoverflow.com/a/70023212/19085332

    line: (xmin, xmax)

    :param dummy: this is just a placeholder
    :type dummy: int
    """
    return line1[0] <= line2[1] - tol and line2[0] <= line1[1] - tol


def is_overlapping_2D(box1, box2, tol=0):
    """
    Based on https://stackoverflow.com/a/70023212/19085332

    box: (xmin, ymin, xmax, ymax)

    :param dummy: this is just a placeholder
    :type dummy: int
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
    Based on https://stackoverflow.com/a/70023212/19085332

    box: (xmin, ymin, zmin, xmax, ymax, zmax)

    :param dummy: this is just a placeholder
    :type dummy: int
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


def get_overlapping_pair(tmp_df, tol=0):
    """
    Description

    :param dummy: this is just a placeholder
    :type dummy: int
    """
    # NOTE: here we use positional indices (i.e. starting from 0)
    num_lines = len(tmp_df.index)
    for pos_ind_1 in range(num_lines):
        for pos_ind_2 in range(pos_ind_1):
            if is_overlapping_2D(
                tmp_df.iloc[pos_ind_1], tmp_df.iloc[pos_ind_2], tol=tol
            ):
                return (pos_ind_1, pos_ind_2)
    return False


def get_overlapping_pairs_3D(tmp_df, pixel_sizes):
    """
    Description

    :param dummy: this is just a placeholder
    :type dummy: int
    """
    # NOTE: here we use positional indices (i.e. starting from 0)
    tol = 10e-10
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


def remove_FOV_overlaps(df):
    """
    Description

    :param dummy: this is just a placeholder
    :type dummy: int
    """

    # Set tolerance (this should be much smaller than pixel size or expected
    # round-offs), and maximum number of iterations in constraint solver
    tol = 1e-10
    max_iterations = 200

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

        # NOTE: these are positional indices (i.e. starting from 0)
        pair_pos_indices = get_overlapping_pair(
            df.loc[well][list_columns], tol=tol
        )

        # Keep going until there are no overlaps, or until iteration reaches
        # max_iterations
        iteration = 0
        while pair_pos_indices:
            iteration += 1

            # Identify overlapping FOVs
            pos_ind_1, pos_ind_2 = pair_pos_indices
            fov_id_1 = df.loc[well].index[pos_ind_1]
            fov_id_2 = df.loc[well].index[pos_ind_2]
            xmin_1, ymin_1, xmax_1, ymax_1 = df.loc[well][list_columns].iloc[
                pos_ind_1
            ]
            xmin_2, ymin_2, xmax_2, ymax_2 = df.loc[well][list_columns].iloc[
                pos_ind_2
            ]

            logger.info(
                f"{iteration=}, removing overlap between"
                f" {fov_id_1=} and {fov_id_2=}"
            )

            # Two overlapping FOVs MUST share either a vertical boundary or an
            # horizontal one
            is_x_equal = abs(xmin_1 - xmin_2) < tol and (xmax_1 - xmax_2) < tol
            is_y_equal = abs(ymin_1 - ymin_2) < tol and (ymax_1 - ymax_2) < tol
            if is_x_equal + is_y_equal != 1:
                raise Exception(
                    "Two overlapping FOVs MUST share either a "
                    "vertical boundary or an horizontal one, but "
                    f"{is_x_equal=} and {is_y_equal=}"
                )

            # Compute and apply shift (either along x or y)
            if is_y_equal:
                min_max_x = min(xmax_1, xmax_2)
                max_min_x = max(xmin_1, xmin_2)
                shift_x = min_max_x - max_min_x
                ind = df.loc[well].loc[:, "xmin"] >= max_min_x - tol
                if not (shift_x > 0.0 and ind.to_numpy().max() > 0):
                    raise Exception(
                        "Something wrong in remove_FOV_overlaps\n"
                        f"{shift_x=}\n"
                        f"{ind.to_numpy()=}"
                    )
                df.loc[well].loc[ind, "xmin"] += shift_x
                df.loc[well].loc[ind, "xmax"] += shift_x
                df.loc[well].loc[ind, "x_micrometer"] += shift_x
            if is_x_equal:
                min_max_y = min(ymax_1, ymax_2)
                max_min_y = max(ymin_1, ymin_2)
                shift_y = min_max_y - max_min_y
                ind = df.loc[well].loc[:, "ymin"] >= max_min_y - tol
                if not (shift_y > 0.0 and ind.to_numpy().max() > 0):
                    raise Exception(
                        "Something wrong in remove_FOV_overlaps\n"
                        f"{shift_y=}\n"
                        f"{ind.to_numpy()=}"
                    )
                df.loc[well].loc[ind, "ymin"] += shift_y
                df.loc[well].loc[ind, "ymax"] += shift_y
                df.loc[well].loc[ind, "y_micrometer"] += shift_y

            pair_pos_indices = get_overlapping_pair(
                df.loc[well][list_columns], tol=tol
            )
            if iteration > max_iterations:
                raise Exception("Something went wrong, {num_iteration=}")
    df.drop(list_columns, axis=1, inplace=True)

    return df

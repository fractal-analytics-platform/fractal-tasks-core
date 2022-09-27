import sys

import numpy as np
import pandas as pd


def is_overlapping_1D(line1, line2, tol=0):
    """
    Based on https://stackoverflow.com/a/70023212/19085332

    line:
        (xmin, xmax)
    """
    return line1[0] <= line2[1] - tol and line2[0] <= line1[1] - tol


def is_overlapping_2D(box1, box2, tol=0):
    """
    Based on https://stackoverflow.com/a/70023212/19085332

    box:
        (xmin, ymin, xmax, ymax)
    """
    overlap_x = is_overlapping_1D(
        [box1[0], box1[2]], [box2[0], box2[2]], tol=tol
    )
    overlap_y = is_overlapping_1D(
        [box1[1], box1[3]], [box2[1], box2[3]], tol=tol
    )
    return overlap_x and overlap_y


def get_box(xmin, ymin, xmax, ymax, line):
    return xmin[line], ymin[line], xmax[line], ymax[line]


def get_overlapping_pair(xmin, xmax, ymin, ymax, tol=0):
    num_lines = len(xmin)
    for line_1 in range(num_lines):
        box_1 = get_box(xmin, ymin, xmax, ymax, line_1)
        for line_2 in range(line_1):
            box_2 = get_box(xmin, ymin, xmax, ymax, line_2)
            if is_overlapping_2D(box_1, box_2, tol=tol):
                return (line_1, line_2)
    return False


# Load file
csv_file = sys.argv[1]
big_df = pd.read_csv(csv_file, index_col=[0, 1])

# Set tolerance (this should be much smaller than pixel size or expected
# round-offs), and maximum number of iterations in constraint solver
tol = 1e-8
max_iterations = 200

# Select first well
wells = sorted(list(set([ind[0] for ind in big_df.index])))
for well in wells:

    print(f"Start analyzing {well=}")
    df = big_df.loc[well]
    xmin = np.array(df.loc[:, "xmin"])
    ymin = np.array(df.loc[:, "ymin"])
    xmax = np.array(df.loc[:, "xmax"])
    ymax = np.array(df.loc[:, "ymax"])

    pair = get_overlapping_pair(xmin, xmax, ymin, ymax, tol=tol)

    iteration = 0
    while pair:
        iteration += 1
        line_1, line_2 = pair
        print(
            f"({iteration}) Resolving overlap between FOVs {line_1} and {line_2}"
        )

        min_x_1, max_x_1 = [a[line_1] for a in [xmin, xmax]]
        min_y_1, max_y_1 = [a[line_1] for a in [ymin, ymax]]
        min_x_2, max_x_2 = [a[line_2] for a in [xmin, xmax]]
        min_y_2, max_y_2 = [a[line_2] for a in [ymin, ymax]]

        is_x_equal = abs(min_x_1 - min_x_2) < tol and (max_x_1 - max_x_2) < tol
        is_y_equal = abs(min_y_1 - min_y_2) < tol and (max_y_1 - max_y_2) < tol
        assert is_x_equal + is_y_equal == 1
        if is_x_equal:
            min_max_y = min(max_y_1, max_y_2)
            max_min_y = max(min_y_1, min_y_2)
            shift_y = min_max_y - max_min_y
            assert shift_y > 0.0
            ind = ymin >= max_min_y - tol
            assert ind.sum() > 0
            ymin[ind] += shift_y
            ymax[ind] += shift_y
        if is_y_equal:
            min_max_x = min(max_x_1, max_x_2)
            max_min_x = max(min_x_1, min_x_2)
            shift_x = min_max_x - max_min_x
            assert shift_x > 0.0
            ind = xmin >= max_min_x - tol
            assert ind.sum() > 0
            xmin[ind] += shift_x
            xmax[ind] += shift_x

        pair = get_overlapping_pair(xmin, xmax, ymin, ymax, tol=tol)
        if iteration > max_iterations:
            raise Exception("Something went wrong, {num_iteration=}")
    print(f"End of {well=}")
    print()

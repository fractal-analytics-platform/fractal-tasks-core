# Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Tommaso Comparin <tommaso.comparin@exact-lab.it>
# Joel Lüthi  <joel.luethi@fmi.ch>
#
# This file is part of Fractal and was originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
"""
Functions to identify overlaps between regions, not related to table specs.
"""
import logging
from typing import Sequence

logger = logging.getLogger(__name__)


def is_overlapping_1D(
    line1: Sequence[float], line2: Sequence[float], tol: float = 1e-10
) -> bool:
    """
    Given two intervals, finds whether they overlap.

    This is based on https://stackoverflow.com/a/70023212/19085332, and we
    additionally use a finite tolerance for floating-point comparisons.

    Args:
        line1: The boundaries of the first interval, written as
            `[x_min, x_max]`.
        line2: The boundaries of the second interval, written as
            `[x_min, x_max]`.
        tol: Finite tolerance for floating-point comparisons.
    """
    return line1[0] <= line2[1] - tol and line2[0] <= line1[1] - tol


def is_overlapping_2D(
    box1: Sequence[float], box2: Sequence[float], tol: float = 1e-10
) -> bool:
    """
    Given two rectangular boxes, finds whether they overlap.

    This is based on https://stackoverflow.com/a/70023212/19085332, and we
    additionally use a finite tolerance for floating-point comparisons.

    Args:
        box1: The boundaries of the first rectangle, written as
            `[x_min, y_min, x_max, y_max]`.
        box2: The boundaries of the second rectangle, written as
            `[x_min, y_min, x_max, y_max]`.
        tol: Finite tolerance for floating-point comparisons.
    """
    overlap_x = is_overlapping_1D(
        [box1[0], box1[2]], [box2[0], box2[2]], tol=tol
    )
    overlap_y = is_overlapping_1D(
        [box1[1], box1[3]], [box2[1], box2[3]], tol=tol
    )
    return overlap_x and overlap_y


def is_overlapping_3D(
    box1: Sequence[float], box2: Sequence[float], tol: float = 1e-10
) -> bool:
    """
    Given two three-dimensional boxes, finds whether they overlap.

    This is based on https://stackoverflow.com/a/70023212/19085332, and we
    additionally use a finite tolerance for floating-point comparisons.

    Args:
        box1: The boundaries of the first box, written as
            `[x_min, y_min, z_min, x_max, y_max, z_max]`.
        box2: The boundaries of the second box, written as
            `[x_min, y_min, z_min, x_max, y_max, z_max]`.
        tol: Finite tolerance for floating-point comparisons.
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


def _is_overlapping_1D_int(
    line1: Sequence[int],
    line2: Sequence[int],
) -> bool:
    """
    Given two integer intervals, find whether they overlap

    This is the same as `is_overlapping_1D` (based on
    https://stackoverflow.com/a/70023212/19085332), for integer-valued
    intervals.

    Args:
        line1: The boundaries of the first interval , written as
            `[x_min, x_max]`.
        line2: The boundaries of the second interval , written as
            `[x_min, x_max]`.
    """
    return line1[0] < line2[1] and line2[0] < line1[1]


def _is_overlapping_3D_int(box1: list[int], box2: list[int]) -> bool:
    """
    Given two three-dimensional integer boxes, find whether they overlap.

    This is the same as is_overlapping_3D (based on
    https://stackoverflow.com/a/70023212/19085332), for integer-valued
    boxes.

    Args:
        box1: The boundaries of the first box, written as
            `[x_min, y_min, z_min, x_max, y_max, z_max]`.
        box2: The boundaries of the second box, written as
            `[x_min, y_min, z_min, x_max, y_max, z_max]`.
    """
    overlap_x = _is_overlapping_1D_int([box1[0], box1[3]], [box2[0], box2[3]])
    overlap_y = _is_overlapping_1D_int([box1[1], box1[4]], [box2[1], box2[4]])
    overlap_z = _is_overlapping_1D_int([box1[2], box1[5]], [box2[2], box2[5]])
    return overlap_x and overlap_y and overlap_z

"""
Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
University of Zurich

Original authors:
Tommaso Comparin <tommaso.comparin@exact-lab.it>
Jacopo Nespolo <jacopo.nespolo@exact-lab.it>

This file is part of Fractal and was originally developed by eXact lab S.r.l.
<exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
Institute for Biomedical Research and Pelkmans Lab from the University of
Zurich.
"""
import pytest

from fractal_tasks_core.cellvoyager.wells import generate_row_col_split
from fractal_tasks_core.cellvoyager.wells import get_filename_well_id

params_well_id = [
    ("A", "01", "A01"),
    ("H", "06", "H06"),
    ("Aa", "011", "A01.a1"),
    ("Bc", "023", "B02.c3"),
]


@pytest.mark.parametrize("row,col,expected", params_well_id)
def test_get_filename_well_id(row, col, expected):
    assert get_filename_well_id(row, col) == expected


def test_not_implemented_filename_well_id():
    with pytest.raises(NotImplementedError):
        get_filename_well_id(row="AAC", col="01234")


params_row_col_split = [
    (
        ["B03", "B05", "B07", "A01", "H08"],
        [("A", "01"), ("B", "03"), ("B", "05"), ("B", "07"), ("H", "08")],
    ),
    (
        ["B03.d1", "B05.b1", "B07.c4", "B07.c3", "B07.a4", "A01.a3", "H08.d2"],
        # Sorting goes by new row name first, then column
        [
            ("Aa", "013"),
            ("Ba", "074"),
            ("Bb", "051"),
            ("Bc", "073"),
            ("Bc", "074"),
            ("Bd", "031"),
            ("Hd", "082"),
        ],
    ),
]


@pytest.mark.parametrize("wells,expected_result", params_row_col_split)
def test_generate_row_col_split(wells, expected_result):
    assert generate_row_col_split(wells) == expected_result


def test_not_implemented_row_col_split():
    wells = ["A011"]
    with pytest.raises(NotImplementedError):
        generate_row_col_split(wells)
    wells = ["A01.a01"]
    with pytest.raises(NotImplementedError):
        generate_row_col_split(wells)

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

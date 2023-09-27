# Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Tommaso Comparin <tommaso.comparin@exact-lab.it>
# Marco Franzon <marco.franzon@exact-lab.it>
#
# This file is part of Fractal and was originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
"""
Helper functions for operations on OME-NGFF metadata.
"""
import logging
from pathlib import Path
from typing import Any

import zarr


logger = logging.getLogger(__name__)


def rescale_datasets(
    *,
    datasets: list[dict],
    coarsening_xy: int,
    reference_level: int,
    remove_channel_axis: bool = False,
) -> list[dict]:
    """
    Given a set of datasets (as per OME-NGFF specs), update their "scale"
    transformations in the YX directions by including a prefactor
    (coarsening_xy**reference_level).

    Args:
        datasets: list of datasets (as per OME-NGFF specs).
        coarsening_xy: linear coarsening factor between subsequent levels.
        reference_level: TBD
        remove_channel_axis: If `True`, remove the first item of all `scale`
            transformations.
    """

    # Construct rescaled datasets
    new_datasets = []
    for ds in datasets:
        new_ds = {}

        # Copy all keys that are not coordinateTransformations (e.g. path)
        for key in ds.keys():
            if key != "coordinateTransformations":
                new_ds[key] = ds[key]

        # Update coordinateTransformations
        old_transformations = ds["coordinateTransformations"]
        new_transformations = []
        for t in old_transformations:
            if t["type"] == "scale":
                new_t: dict[str, Any] = t.copy()
                # Rescale last two dimensions (that is, Y and X)
                prefactor = coarsening_xy**reference_level
                new_t["scale"][-2] = new_t["scale"][-2] * prefactor
                new_t["scale"][-1] = new_t["scale"][-1] * prefactor
                if remove_channel_axis:
                    new_t["scale"].pop(0)
                new_transformations.append(new_t)
            else:
                new_transformations.append(t)
        new_ds["coordinateTransformations"] = new_transformations
        new_datasets.append(new_ds)

    return new_datasets


def get_table_path_dict(input_path: Path, component: str) -> dict[str, str]:
    """
    Compile dictionary of (table name, table path) key/value pairs.


    Args:
        input_path:
            Path to the parent folder of a plate zarr group (e.g.
            `/some/path/`).
        component:
            Path (relative to `input_path`) to an image zarr group (e.g.
            `plate.zarr/B/03/0`).

    Returns:
        Dictionary with table names as keys and table paths as values. If
            `tables` Zarr group is missing, or if it does not have a `tables`
            key, then return an empty dictionary.
    """

    try:
        tables_group = zarr.open_group(f"{input_path / component}/tables", "r")
        table_list = tables_group.attrs["tables"]
    except (zarr.errors.GroupNotFoundError, KeyError):
        table_list = []

    table_path_dict = {}
    for table in table_list:
        table_path_dict[table] = f"{input_path / component}/tables/{table}"

    return table_path_dict

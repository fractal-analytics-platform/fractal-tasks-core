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
Functions to handle `.zattrs` files and their contents.
"""
import json
import logging
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


def extract_zyx_pixel_sizes(zattrs_path: str, level: int = 0) -> list[float]:
    """
    Load multiscales/datasets from `.zattrs` file and read the pixel sizes for
    a given resoluion level.

    Args:
        zattrs_path: Path to `.zattrs` file.
        level: Resolution level for which the pixe sizes are required.

    Returns:
        ZYX pixel sizes.
    """

    with open(zattrs_path, "r") as jsonfile:
        zattrs = json.load(jsonfile)

    try:

        # Identify multiscales
        multiscales = zattrs["multiscales"]

        # Check that there is a single multiscale
        if len(multiscales) > 1:
            raise ValueError(
                f"ERROR: There are {len(multiscales)} multiscales"
            )

        # Check that Z axis is present, raise a warning otherwise
        axes = [ax["name"] for ax in multiscales[0]["axes"]]
        if "z" not in axes:
            logger.warning(
                f"Z axis is not present in {axes=}. This case may work "
                "by accident, but it is not fully supported."
            )

        # Check that there are no datasets-global transformations
        if "coordinateTransformations" in multiscales[0].keys():
            raise NotImplementedError(
                "Global coordinateTransformations at the multiscales "
                "level are not currently supported"
            )

        # Identify all datasets (AKA pyramid levels)
        datasets = multiscales[0]["datasets"]

        # Select highest-resolution dataset
        transformations = datasets[level]["coordinateTransformations"]
        for t in transformations:
            if t["type"] == "scale":
                # FIXME: Using [-3:] indices is a hack to deal with the fact
                # that the coordinationTransformation can contain additional
                # entries (e.g. scaling for the channels)
                # https://github.com/fractal-analytics-platform/fractal-tasks-core/issues/420
                pixel_sizes = t["scale"][-3:]
                if min(pixel_sizes) < 1e-9:
                    raise ValueError(
                        f"pixel_sizes in {zattrs_path} are {pixel_sizes}"
                    )
                return pixel_sizes

        raise ValueError(
            f"No scale transformation found for level {level} in {zattrs_path}"
        )

    except KeyError as e:
        raise KeyError(
            f"extract_zyx_pixel_sizes_from_zattrs failed, for {zattrs_path}\n",
            e,
        )


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


def get_acquisition_paths(zattrs: dict) -> dict[int, str]:
    """
    Create mapping from acquisition indices to corresponding paths.

    Runs on the well .zattrs content and loads the relative paths in the well.

    Args:
        zattrs:
            Attributes of a well zarr group.

    Returns:
        Dictionary with `(acquisition index: image path)` key/value pairs.
    """
    acquisition_dict = {}
    for image in zattrs["well"]["images"]:
        if "acquisition" not in image:
            raise ValueError(
                "Cannot get acquisition paths for Zarr files without "
                "'acquisition' metadata at the well level"
            )
        if image["acquisition"] in acquisition_dict:
            raise NotImplementedError(
                "This task is not implemented for wells with multiple images "
                "of the same acquisition"
            )
        acquisition_dict[image["acquisition"]] = image["path"]
    return acquisition_dict


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
        Dictionary with table names as keys and table paths as values.
    """

    try:
        with open(f"{input_path / component}/tables/.zattrs", "r") as f_zattrs:
            table_list = json.load(f_zattrs)["tables"]
    except FileNotFoundError:
        table_list = []

    table_path_dict = {}
    for table in table_list:
        table_path_dict[table] = f"{input_path / component}/tables/{table}"

    return table_path_dict


def get_axes_names(attrs: dict) -> list:
    """
    Get the axes names of a .zattrs dictionary

    .zattrs dicts usually contain their axes in the multiscales metadata.
    This function returns a list of the axes names in the order they appeared
    in the metadata.

    Args:
        attrs: The .zattrs group of an OME-Zarr image as a dict

    Returns:
        List of access names
    """
    try:
        axes = attrs["multiscales"][0]["axes"]
    except (KeyError, TypeError) as e:
        raise ValueError(
            f"{attrs=} does not contain the necessary information to get "
            f"axes, raising an exception {e=}"
        )
    names = []
    for ax in axes:
        names.append(ax["name"])
    return names

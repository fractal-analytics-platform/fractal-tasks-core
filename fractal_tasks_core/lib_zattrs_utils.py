"""
Copyright 2022 (C)
    Friedrich Miescher Institute for Biomedical Research and
    University of Zurich

    Original authors:
    Tommaso Comparin <tommaso.comparin@exact-lab.it>
    Marco Franzon <marco.franzon@exact-lab.it>

    This file is part of Fractal and was originally developed by eXact lab
    S.r.l.  <exact-lab.it> under contract with Liberali Lab from the Friedrich
    Miescher Institute for Biomedical Research and Pelkmans Lab from the
    University of Zurich.

Functions to handle .zattrs files and their contents
"""
import json
from typing import Any
from typing import Dict
from typing import List


def extract_zyx_pixel_sizes(zattrs_path: str, level: int = 0) -> List[float]:
    """
    Load multiscales/datasets from .zattrs file and read the pixel sizes for a
    given resoluion level.

    :param zattrs_path: Path to .zattrs file
    :param level: Resolution level for which the pixe sizes are required
    :returns: ZYX pixel sizes
    """

    with open(zattrs_path, "r") as jsonfile:
        zattrs = json.load(jsonfile)

    try:

        # Identify multiscales
        multiscales = zattrs["multiscales"]

        # Check that there is a single multiscale
        if len(multiscales) > 1:
            raise Exception(f"ERROR: There are {len(multiscales)} multiscales")

        # Check that there are no datasets-global transformations
        if "coordinateTransformations" in multiscales[0].keys():
            raise NotImplementedError(
                "global coordinateTransformations at the multiscales "
                "level are not currently supported"
            )

        # Identify all datasets (AKA pyramid levels)
        datasets = multiscales[0]["datasets"]

        # Select highest-resolution dataset
        transformations = datasets[level]["coordinateTransformations"]
        for t in transformations:
            if t["type"] == "scale":
                pixel_sizes = t["scale"]
                if min(pixel_sizes) < 1e-9:
                    raise Exception(
                        f"ERROR: pixel_sizes in {zattrs_path} are", pixel_sizes
                    )
                return pixel_sizes

        raise Exception(
            "ERROR:"
            f" no scale transformation found for level {level}"
            f" in {zattrs_path}"
        )

    except KeyError as e:
        raise KeyError(
            "extract_zyx_pixel_sizes_from_zattrs failed, for {zattrs_path}\n",
            e,
        )


def rescale_datasets(
    *,
    datasets: List[Dict],
    coarsening_xy: int,
    reference_level: int,
) -> List[Dict]:
    """
    Given a set of datasets (as per OME-NGFF specs), update their "scale"
    transformations in the YX directions by including a prefactor
    (coarsening_xy**reference_level).

    :param datasets: list of datasets (as per OME-NGFF specs)
    :param coarsening_xy: linear coarsening factor between subsequent levels
    :param reference_level: TBD
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
                new_t: Dict[str, Any] = {"type": "scale"}
                new_t["scale"] = [
                    t["scale"][0],
                    t["scale"][1] * coarsening_xy**reference_level,
                    t["scale"][2] * coarsening_xy**reference_level,
                ]
                new_transformations.append(new_t)
            else:
                new_transformations.append(t)
        new_ds["coordinateTransformations"] = new_transformations
        new_datasets.append(new_ds)

    return new_datasets

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
Helper functions for operations on Zarr attributes and OME-NGFF metadata.
"""
import logging
from pathlib import Path
from typing import Any
from typing import Sequence
from typing import Union

import zarr

from fractal_tasks_core.ngff.zarr_utils import load_NgffWellMeta


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


def _get_table_path_dict(zarr_url: str) -> dict[str, str]:
    """
    Compile dictionary of (table name, table path) key/value pairs.


    Args:
        zarr_url:
            Path or url to the individual OME-Zarr image to be processed.

    Returns:
        Dictionary with table names as keys and table paths as values. If
            `tables` Zarr group is missing, or if it does not have a `tables`
            key, then return an empty dictionary.
    """

    try:
        tables_group = zarr.open_group(f"{zarr_url}/tables", "r")
        table_list = tables_group.attrs["tables"]
    except (zarr.errors.GroupNotFoundError, KeyError):
        table_list = []

    table_path_dict = {}
    for table in table_list:
        table_path_dict[table] = f"{zarr_url}/tables/{table}"

    return table_path_dict


def _find_omengff_acquisition(image_zarr_path: Path) -> Union[int, None]:
    """
    Discover the acquisition index based on OME-NGFF metadata.

    Given the path to a zarr image folder (e.g. `/path/plate.zarr/B/03/0`),
    extract the acquisition index from the `.zattrs` file of the parent
    folder (i.e. at the well level), or return `None` if acquisition is not
    specified.

    Notes:

    1. For non-multiplexing datasets, acquisition is not a required
       information in the metadata. If it is not there, this function
       returns `None`.
    2. This function fails if we use an image that does not belong to
       an OME-NGFF well.

    Args:
        image_zarr_path: Full path to an OME-NGFF image folder.
    """

    # Identify well path and attrs
    well_zarr_path = image_zarr_path.parent
    if not (well_zarr_path / ".zattrs").exists():
        raise ValueError(
            f"{str(well_zarr_path)} must be an OME-NGFF well "
            "folder, but it does not include a .zattrs file."
        )
    well_group = zarr.open_group(str(well_zarr_path))
    attrs_images = well_group.attrs["well"]["images"]

    # Loook for the acquisition of the current image (if any)
    acquisition = None
    for img_dict in attrs_images:
        if (
            img_dict["path"] == image_zarr_path.name
            and "acquisition" in img_dict.keys()
        ):
            acquisition = img_dict["acquisition"]
            break

    return acquisition


def get_parameters_from_metadata(
    *,
    keys: Sequence[str],
    metadata: dict[str, Any],
    image_zarr_path: Path,
) -> dict[str, Any]:
    """
    Flexibly extract parameters from metadata dictionary

    This covers both parameters which are acquisition-specific (if the image
    belongs to an OME-NGFF array and its acquisition is specified) or simply
    available in the dictionary.
    The two cases are handled as:
    ```
    metadata[acquisition]["some_parameter"]  # acquisition available
    metadata["some_parameter"]               # acquisition not available
    ```

    Args:
        keys: list of required parameters.
        metadata: metadata dictionary.
        image_zarr_path: full path to image, e.g. `/path/plate.zarr/B/03/0`.
    """

    parameters = {}
    acquisition = _find_omengff_acquisition(image_zarr_path)
    if acquisition is not None:
        parameters["acquisition"] = acquisition

    for key in keys:
        if acquisition is None:
            parameter = metadata[key]
        else:
            try:
                parameter = metadata[key][str(acquisition)]
            except TypeError:
                parameter = metadata[key]
            except KeyError:
                parameter = metadata[key]
        parameters[key] = parameter
    return parameters


def create_well_acquisition_dict(
    zarr_urls: list[str],
) -> dict[str, dict[int, str]]:
    """
    Parses zarr_urls & groups them by HCS wells & acquisition

    Generates a dict with keys a unique description of the acquisition
    (e.g. plate + well for HCS plates). The values are dictionaries. The keys
    of the secondary dictionary are the acqusitions, its values the `zarr_url`
    for a given acquisition.

    Args:
        zarr_urls: List of zarr_urls

    Returns:
        image_groups
    """
    image_groups = dict()

    # Dict to cache well-level metadata
    well_metadata = dict()
    for zarr_url in zarr_urls:
        well_path, img_sub_path = _split_well_path_image_path(zarr_url)
        # For the first zarr_url of a well, load the well metadata and
        # initialize the image_groups dict
        if well_path not in image_groups:
            well_meta = load_NgffWellMeta(well_path)
            well_metadata[well_path] = well_meta.well
            image_groups[well_path] = {}

        # For every zarr_url, add it under the well_path & acquisition keys to
        # the image_groups dict
        for image in well_metadata[well_path].images:
            if image.path == img_sub_path:
                if image.acquisition in image_groups[well_path]:
                    raise ValueError(
                        "This task has not been built for OME-Zarr HCS plates"
                        "with multiple images of the same acquisition per well"
                        f". {image.acquisition} is the acquisition for "
                        f"multiple images in {well_path=}."
                    )

                image_groups[well_path][image.acquisition] = zarr_url
    return image_groups


def _split_well_path_image_path(zarr_url: str) -> tuple[str, str]:
    """
    Returns path to well folder for HCS OME-Zarr `zarr_url`.
    """
    zarr_url = zarr_url.rstrip("/")
    well_path = "/".join(zarr_url.split("/")[:-1])
    img_path = zarr_url.split("/")[-1]
    return well_path, img_path

# Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Tommaso Comparin <tommaso.comparin@exact-lab.it>
# Joel Lüthi <joel.luethi@uzh.ch>
#
# This file is part of Fractal and was originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
"""
Initializes the parallelization list for registration in HCS plates.
"""
import logging
from typing import Any

import zarr
from pydantic.decorator import validate_arguments

from fractal_tasks_core.ngff.specs import Well


logger = logging.getLogger(__name__)


def _split_well_path_image_path(zarr_url: str):
    """
    Returns path to well folder for HCS OME-Zarr zarr_url
    """
    zarr_url = zarr_url.rstrip("/")
    well_path = "/".join(zarr_url.split("/")[:-1])
    img_path = zarr_url.split("/")[-1]
    return well_path, img_path


@validate_arguments
def image_based_registration_hcs_init(
    *,
    # Fractal arguments
    zarr_urls: list[str],
    zarr_dir: str,
    # Task-specific arguments
    reference_cycle: int = 0,
) -> dict[str, Any]:
    """
    Initialized calculate registration task

    This task prepares a parallelization list of all zarr_urls that need to be
    used to calculate the registration between cycles (all zarr_urls except
    the reference cycle vs. the reference cycle).
    This task only works for HCS OME-Zarrs for 2 reasons: Only HCS OME-Zarrs
    currently have defined acquisition metadata to determine reference cycles.
    And we have only implemented the grouping of images for HCS OME-Zarrs by
    well (with the assumption that every well just has 1 image per acqusition).

    Args:
        zarr_urls: List of paths or urls to the individual OME-Zarr image to
            be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        zarr_dir: path of the directory where the new OME-Zarrs will be
            created. Not used by this task.
            (standard argument for Fractal tasks, managed by Fractal server).
        reference_cycle: Which cycle to register against. Needs to match the
            acquisition metadata in the OME-Zarr image.

    Returns:
        task_output: Dictionary for Fractal server that contains a
            parallelization list.
    """
    logger.info(
        f"Running `image_based_registration_hcs_init` for {zarr_urls=}"
    )
    # Dict with keys a unique description of the acquisition (e.g. plate +
    # well for HCS plates). The values are a dictionary. The keys of the
    # secondary dictionary are the acqusitions, its values the zarr_url for
    # a given acquisition
    image_groups = dict()
    # Dict to cache well-level metadata
    well_metadata = dict()
    for zarr_url in zarr_urls:
        well_path, img_sub_path = _split_well_path_image_path(zarr_url)
        # For the first zarr_url of a well, load the well metadata and
        # initialize the image_groups dict
        if well_path not in image_groups:
            image_groups[well_path] = {}
            well_group = zarr.open_group(well_path, mode="r")
            well_metadata[well_path] = Well(**well_group.attrs.asdict())

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

    # Create the parallelization list
    parallelization_list = []
    for key, image_group in image_groups.items():
        # Assert that all image groups have the reference cycle present
        if reference_cycle not in image_group.keys():
            raise ValueError(
                f"Registration with {reference_cycle=} can only work if all"
                "wells have the reference cycle present. It was not found"
                f"for well {key}."
            )
        # Add all zarr_urls except the reference cycle to the
        # parallelization list
        for acquisition, zarr_url in image_group.items():
            if acquisition != reference_cycle:
                reference_zarr_url = image_group[reference_cycle]
                parallelization_list.append(
                    dict(
                        zarr_url=zarr_url,
                        init_args=dict(reference_zarr_url=reference_zarr_url),
                    )
                )

    return dict(parallelization_list=parallelization_list)


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=image_based_registration_hcs_init,
        logger_name=logger.name,
    )

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
Applies the multiplexing translation to all ROI tables
"""
import logging

from pydantic.decorator import validate_arguments

from fractal_tasks_core.tasks._registration_utils import (
    create_well_acquisition_dict,
)

logger = logging.getLogger(__name__)


@validate_arguments
def init_group_by_well_for_multiplexing(
    *,
    # Fractal arguments
    zarr_urls: list[str],
    zarr_dir: str,
    # Task-specific arguments
    reference_cycle: int = 0,
) -> dict[str, list[str]]:
    """
    Finds images for all acquisitions per well.

    Returns the parallelization_list to run `find_registration_consensus`.

    Args:
        zarr_urls: List of paths or urls to the individual OME-Zarr image to
            be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        zarr_dir: path of the directory where the new OME-Zarrs will be
            created. Not used by this task.
            (standard argument for Fractal tasks, managed by Fractal server).
        reference_cycle: Which cycle to register against. Defaults to 0,
            which is the first OME-Zarr image in the well, usually the first
            cycle that was provided
    """
    logger.info(
        f"Running `init_group_by_well_for_multiplexing` for {zarr_urls=}"
    )
    image_groups = create_well_acquisition_dict(zarr_urls)

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

        # Create a parallelization list entry for each image group
        zarr_url_list = []
        for acquisition, zarr_url in image_group.items():
            if acquisition == reference_cycle:
                reference_zarr_url = zarr_url

            zarr_url_list.append(zarr_url)

        parallelization_list.append(
            dict(
                zarr_url=reference_zarr_url,
                init_args=dict(zarr_url_list=zarr_url_list),
            )
        )

    return dict(parallelization_list=parallelization_list)


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=init_group_by_well_for_multiplexing,
        logger_name=logger.name,
    )

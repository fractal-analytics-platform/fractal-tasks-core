# Copyright 2022-2026 (C) BioVisionCenter, University of Zurich
"""
Initializes the parallelization list for registration in HCS plates.
"""

import logging
from typing import Any

from ngio import open_ome_zarr_well
from pydantic import validate_call

from fractal_tasks_core.tasks._plate_utils import group_by_well

logger = logging.getLogger("image_based_registration_hcs_init")


@validate_call
def image_based_registration_hcs_init(
    *,
    # Fractal parameters
    zarr_urls: list[str],
    zarr_dir: str,
    # Core parameters
    reference_acquisition: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    """
    Initialized calculate registration task

    This task prepares a parallelization list of all zarr_urls that need to be
    used to calculate the registration between acquisitions (all zarr_urls
    except the reference acquisition vs. the reference acquisition).
    This task only works for HCS OME-Zarrs for 2 reasons: Only HCS OME-Zarrs
    currently have defined acquisition metadata to determine reference
    acquisitions. And we have only implemented the grouping of images for
    HCS OME-Zarrs by well (with the assumption that every well just has 1
    image per acqusition).

    Args:
        zarr_urls: List of paths or urls to the individual OME-Zarr image to
            be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        zarr_dir: path of the directory where the new OME-Zarrs will be
            created. Not used by this task.
            (standard argument for Fractal tasks, managed by Fractal server).
        reference_acquisition: Which acquisition to register against. Needs to
            match the acquisition metadata in the OME-Zarr image.

    Returns:
        task_output: Dictionary for Fractal server that contains a
            parallelization list.
    """
    logger.info(f"Running `image_based_registration_hcs_init` for {zarr_urls=}")
    wells = group_by_well(zarr_urls)

    parallelization_list = []
    for well_url, well_image_urls in wells.items():
        well = open_ome_zarr_well(well_url)
        logger.info(f"Found well {well} with {len(well_image_urls)} urls.")
        # Find the reference acquisition url for this well
        ref_image_zarr_path = well.paths(acquisition=reference_acquisition)
        # Find the matching image url in the well_image_urls
        # (there should be exactly one)
        ref_image_zarr_url = [
            url.zarr_url
            for url in well_image_urls
            if url.image_path in ref_image_zarr_path
        ]
        if len(ref_image_zarr_url) == 0:
            raise ValueError(
                f"No reference acquisition found for well {well}. "
                f"Expected to find acquisition {reference_acquisition} in the "
                "metadata of the OME-Zarr image, but it was not found."
            )
        elif len(ref_image_zarr_url) > 1:
            raise ValueError(
                f"Multiple reference acquisitions found for well {well}. "
                f"Expected to find exactly one acquisition {reference_acquisition} "
                "in the metadata of the OME-Zarr image, but multiple were found: "
                f"{ref_image_zarr_url}"
            )
        logger.info(
            f"Found reference acquisition for well {well}: {ref_image_zarr_url[0]}"
        )
        ref_path = ref_image_zarr_url[0]
        for url in well_image_urls:
            if url.zarr_url == ref_path:
                continue
            parallelization_list.append(
                dict(
                    zarr_url=url.zarr_url,
                    init_args=dict(reference_zarr_url=ref_path),
                )
            )
    return dict(parallelization_list=parallelization_list)


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=image_based_registration_hcs_init,
        logger_name=logger.name,
    )

# Copyright 2022-2026 (C) BioVisionCenter, University of Zurich
"""
Prepares the parallelization list for consensus registration.
"""

import logging
from typing import Any

from ngio import open_ome_zarr_well
from pydantic import validate_call

from fractal_tasks_core._utils import group_by_well

logger = logging.getLogger("init_registration_consensus")


@validate_call
def init_registration_consensus(
    *,
    # Fractal parameters
    zarr_urls: list[str],
    zarr_dir: str,
    # Core parameters
    reference_acquisition: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    """
    Prepare the list of images needed to compute a registration consensus.

    Finds all images for each well across all acquisitions and returns the
    information required to run `compute_registration_consensus`.

    Args:
        zarr_urls: List of paths or urls to the individual OME-Zarr image to
            be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        zarr_dir: path of the directory where the new OME-Zarrs will be
            created. Not used by this task.
            (standard argument for Fractal tasks, managed by Fractal server).
        reference_acquisition: Which acquisition to register against. Uses the
            OME-NGFF HCS well metadata acquisition keys to find the reference
            acquisition.
    """
    logger.info(f"Running `init_registration_consensus` for {zarr_urls=}")
    wells = group_by_well(zarr_urls)

    parallelization_list = []
    for well_url, well_image_urls in wells.items():
        well = open_ome_zarr_well(well_url)
        logger.info(f"Found well {well} with {len(well_image_urls)} urls.")
        ref_image_zarr_path = well.paths(acquisition=reference_acquisition)
        ref_image_zarr_url = [
            url.zarr_url
            for url in well_image_urls
            if url.image_path in ref_image_zarr_path
        ]
        if len(ref_image_zarr_url) == 0:
            raise ValueError(
                f"Registration with {reference_acquisition=} can only work if "
                "all wells have the reference acquisition present. It was not "
                f"found for well {well_url}."
            )
        elif len(ref_image_zarr_url) > 1:
            raise ValueError(
                f"Multiple reference acquisitions found for well {well_url}. "
                f"Expected to find exactly one acquisition {reference_acquisition} "
                "in the metadata of the OME-Zarr image, but multiple were found: "
                f"{ref_image_zarr_url}"
            )
        ref_path = ref_image_zarr_url[0]
        zarr_url_list = [ref_path] + [
            url.zarr_url for url in well_image_urls if url.zarr_url != ref_path
        ]
        parallelization_list.append(
            dict(
                zarr_url=ref_path,
                init_args=dict(zarr_url_list=zarr_url_list),
            )
        )

    return dict(parallelization_list=parallelization_list)


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=init_registration_consensus,
        logger_name=logger.name,
    )

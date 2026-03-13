# Copyright 2022-2026 (C) BioVisionCenter, University of Zurich
"""
Task for 3D->2D maximum-intensity projection.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import validate_call

from fractal_tasks_core._projection_utils import DaskProjectionMethod, projection_core

logger = logging.getLogger("projection")


@validate_call
def projection(
    *,
    zarr_url: str,
    method: DaskProjectionMethod = DaskProjectionMethod.MIP,
    overwrite: bool = False,
) -> dict[str, Any]:
    """
    Perform intensity projection along Z axis with a chosen method.

    Note: this task will write the output in a new OM-Zarr file
        in the same location as the input one, with the same name plus
        a suffix indicating the projection method used (e.g. "_MIP" for
        maximum intensity projection).

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
        method: Projection method to be used. Implemented methods are:
            - MIP: Maximum intensity projection
            - MINIP: Minimum intensity projection
            - MEANIP: Mean intensity projection
            - SUMIP: Sum intensity projection
        overwrite: If `True`, overwrite the task output.
    """
    if not zarr_url.endswith(".zarr"):
        raise ValueError(f"The input zarr url must end with .zarr, but got {zarr_url}")
    output_zarr_url = zarr_url.removesuffix(".zarr") + f"_{method.abbreviation}.zarr"
    return projection_core(
        input_zarr_url=zarr_url,
        output_zarr_url=output_zarr_url,
        method=method,
        overwrite=overwrite,
    )


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=projection,
        logger_name=logger.name,
    )

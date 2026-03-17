# Copyright 2022-2026 (C) BioVisionCenter, University of Zurich
"""
Task for 3D->2D maximum-intensity projection.
"""

from __future__ import annotations

from typing import Any

from pydantic import validate_call

from fractal_tasks_core._projection_utils import (
    InitArgsMIP,
    projection_core,
)


@validate_call
def compute_projection_hcs(
    *,
    # Fractal parameters
    zarr_url: str,
    init_args: InitArgsMIP,
) -> dict[str, Any]:
    """
    Perform intensity projection for one image in an HCS plate.

    Uses settings prepared by `init_projection_hcs` and stores the output
    in a new OME-Zarr file within the projected plate.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        init_args: Initialization arguments provided by
            `init_projection_hcs`.
    """
    attributes = {"plate": init_args.new_plate_name}
    return projection_core(
        input_zarr_url=init_args.origin_url,
        output_zarr_url=zarr_url,
        method=init_args.method,
        overwrite=init_args.overwrite,
        attributes=attributes,
    )


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=compute_projection_hcs)

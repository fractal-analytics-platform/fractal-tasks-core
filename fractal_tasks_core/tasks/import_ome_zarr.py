# Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Tommaso Comparin <tommaso.comparin@exact-lab.it>
#
# This file is part of Fractal and was originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
"""
Task to import an OME-Zarr.
"""
import logging
from typing import Any
from typing import Sequence

from pydantic.decorator import validate_arguments

logger = logging.getLogger(__name__)


@validate_arguments
def import_ome_zarr(
    *,
    input_paths: Sequence[str],
    output_path: str,
    metadata: dict[str, Any],
    overwrite: bool = False,
) -> dict[str, Any]:
    """
    Import an OME-Zarr

    Args:
        input_paths: TBD
            (standard argument for Fractal tasks, managed by Fractal server).
        output_path: TBD
            (standard argument for Fractal tasks, managed by Fractal server).
        component: TBD
            (standard argument for Fractal tasks, managed by Fractal server).
        metadata: TBD
            (standard argument for Fractal tasks, managed by Fractal server).
        overwrite: TBD
    """

    # Preliminary checks
    if len(input_paths) > 1:
        raise NotImplementedError

    return {}


if __name__ == "__main__":

    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=import_ome_zarr,
        logger_name=logger.name,
    )

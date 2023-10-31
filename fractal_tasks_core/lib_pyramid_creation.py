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
Construct and write pyramid of lower-resolution levels.
"""
import logging
import pathlib
from typing import Callable
from typing import Optional
from typing import Sequence
from typing import Union

import dask.array as da
import numpy as np

logger = logging.getLogger(__name__)


def build_pyramid(
    *,
    zarrurl: Union[str, pathlib.Path],
    overwrite: bool = False,
    num_levels: int = 2,
    coarsening_xy: int = 2,
    chunksize: Optional[Sequence[int]] = None,
    aggregation_function: Optional[Callable] = None,
) -> None:

    """
    Starting from on-disk highest-resolution data, build and write to disk a
    pyramid with `(num_levels - 1)` coarsened levels.
    This function works for 2D, 3D or 4D arrays.

    Args:
        zarrurl: Path of the image zarr group, not including the
            multiscale-level path (e.g. `"some/path/plate.zarr/B/03/0"`).
        overwrite: Whether to overwrite existing pyramid levels.
        num_levels: Total number of pyramid levels (including 0).
        coarsening_xy: Linear coarsening factor between subsequent levels.
        chunksize: Shape of a single chunk.
        aggregation_function: Function to be used when downsampling.
    """

    # Clean up zarrurl
    zarrurl = str(pathlib.Path(zarrurl))  # FIXME

    # Select full-resolution multiscale level
    zarrurl_highres = f"{zarrurl}/0"
    logger.info(f"[build_pyramid] High-resolution path: {zarrurl_highres}")

    # Lazily load highest-resolution data
    data_highres = da.from_zarr(zarrurl_highres)
    logger.info(f"[build_pyramid] High-resolution data: {str(data_highres)}")

    # Check the number of axes and identify YX dimensions
    ndims = len(data_highres.shape)
    if ndims not in [2, 3, 4]:
        raise ValueError(f"{data_highres.shape=}, ndims not in [2,3,4]")
    y_axis = ndims - 2
    x_axis = ndims - 1

    # Set aggregation_function
    if aggregation_function is None:
        aggregation_function = np.mean

    # Compute and write lower-resolution levels
    previous_level = data_highres
    for ind_level in range(1, num_levels):
        # Verify that coarsening is doable
        if min(previous_level.shape[-2:]) < coarsening_xy:
            raise ValueError(
                f"ERROR: at {ind_level}-th level, "
                f"coarsening_xy={coarsening_xy} "
                f"but previous level has shape {previous_level.shape}"
            )
        # Apply coarsening
        newlevel = da.coarsen(
            aggregation_function,
            previous_level,
            {y_axis: coarsening_xy, x_axis: coarsening_xy},
            trim_excess=True,
        ).astype(data_highres.dtype)

        # Apply rechunking
        if chunksize is None:
            newlevel_rechunked = newlevel
        else:
            newlevel_rechunked = newlevel.rechunk(chunksize)
        logger.info(
            f"[build_pyramid] Level {ind_level} data: "
            f"{str(newlevel_rechunked)}"
        )

        # Write zarr and store output (useful to construct next level)
        previous_level = newlevel_rechunked.to_zarr(
            zarrurl,
            component=f"{ind_level}",
            overwrite=overwrite,
            compute=True,
            return_stored=True,
            write_empty_chunks=False,
            dimension_separator="/",
        )

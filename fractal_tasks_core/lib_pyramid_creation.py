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

Construct and write pyramid of lower-resolution levels
"""
import pathlib
from typing import Callable
from typing import Sequence
from typing import Union

import dask.array as da
import numpy as np


def build_pyramid(
    *,
    zarrurl: Union[str, pathlib.Path],
    overwrite: bool = False,
    num_levels: int = 2,
    coarsening_xy: int = 2,
    chunksize: Sequence[int] = None,
    aggregation_function: Callable = None,
):

    """
    Starting from on-disk highest-resolution data, build and write to disk a
    pyramid with (num_levels-1) coarsened levels.
    This function works for 2D, 3D or 4D arrays.

    Example input:
        zarrurl = "some/path/plate.zarr/B/03/0

    :param zarrurl: path of the zarr group, which must already include level 0
    :param overwrite: whether to overwrite existing pyramid levels
    :param num_levels: total number of pyramid levels (including 0)
    :param coarsening_xy: linear coarsening factor between subsequent levels
    :param chunksize: shape of a single chunk
    :param aggregation_function: function to be used when downsampling
    """

    # Clean up zarrurl
    zarrurl = str(pathlib.Path(zarrurl))  # FIXME
    zarrurl_highres = f"{zarrurl}/0"

    # Lazily load highest-resolution data
    data_highres = da.from_zarr(zarrurl_highres)

    # Check the number of axes and identify YX dimensions
    ndims = len(data_highres.shape)
    if ndims not in [2, 3, 4]:
        raise Exception("{data_highres.shape=}, ndims not in [2,3,4]")
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
            raise Exception(
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

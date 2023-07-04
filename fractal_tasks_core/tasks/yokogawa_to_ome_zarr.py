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

Task that writes image data to an existing OME-NGFF zarr array
"""
import logging
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Sequence

import dask.array as da
import zarr
from anndata import read_zarr
from dask.array.image import imread
from pydantic.decorator import validate_arguments

from fractal_tasks_core.lib_channels import get_omero_channel_list
from fractal_tasks_core.lib_channels import OmeroChannel
from fractal_tasks_core.lib_glob import glob_with_multiple_patterns
from fractal_tasks_core.lib_parse_filename_metadata import parse_filename
from fractal_tasks_core.lib_pyramid_creation import build_pyramid
from fractal_tasks_core.lib_read_fractal_metadata import (
    get_parameters_from_metadata,
)
from fractal_tasks_core.lib_regions_of_interest import (
    convert_ROI_table_to_indices,
)
from fractal_tasks_core.lib_zattrs_utils import extract_zyx_pixel_sizes


logger = logging.getLogger(__name__)


def sort_fun(filename: str):
    """
    sort_fun takes a string (filename of a yokogawa images), extract site and
    z-index metadata and returns them as a list of integers

    :param filename: name of the image file
    """

    filename_metadata = parse_filename(filename)
    site = int(filename_metadata["F"])
    z_index = int(filename_metadata["Z"])
    return [site, z_index]


@validate_arguments
def yokogawa_to_ome_zarr(
    *,
    input_paths: Sequence[str],
    output_path: str,
    component: str,
    metadata: Dict[str, Any],
):
    """
    Convert Yokogawa output (png, tif) to zarr file

    This task is typically run after Create OME-Zarr or
    Create OME-Zarr Multiplexing and populates the empty OME-Zarr files that
    were prepared.

    :param input_paths: List of input paths where the OME-Zarrs.
                        Should point to the parent folder
                        containing one or many OME-Zarr files, not the
                        actual OME-Zarr file.
                        Example: ["/some/path/"]
                        This task only supports a single input path.
                        (standard argument for Fractal tasks,
                        managed by Fractal server)
    :param output_path: Unclear. Should be the same as input_path.
                        (standard argument for Fractal tasks,
                        managed by Fractal server)
    :param component: Path to the OME-Zarr image in the OME-Zarr plate that
                      is processed.
                      Example: "some_plate.zarr/B/03/0"
                      (standard argument for Fractal tasks,
                      managed by Fractal server)
    :param metadata: dictionary containing metadata about the OME-Zarr.
                     This task requires the following elements to be present
                     in the metadata:
                     "original_paths": list of paths that correspond to the
                     ``input_paths`` of the create_ome_zarr task (=> where
                     the microscopy image are stored)
                     "num_levels": int, number of pyramid levels in the image.
                     This determines how many pyramid levels are built for
                     the segmentation.
                     "coarsening_xy": int, coarsening factor in XY of the
                     downsampling when building the pyramid.
                     "image_extension": Filename extension of images (e.g.
                     ``"tif"`` or ``"png"``)
                     "image_glob_patterns": Parameter of ``create_ome_zarr``
                     task. If specified, only parse images with filenames
                     that match with all these patterns.
                     (standard argument for Fractal tasks,
                     managed by Fractal server)
    """

    # Preliminary checks
    if len(input_paths) > 1:
        raise NotImplementedError
    zarrurl = Path(input_paths[0]).as_posix() + f"/{component}"

    parameters = get_parameters_from_metadata(
        keys=[
            "original_paths",
            "num_levels",
            "coarsening_xy",
            "image_extension",
            "image_glob_patterns",
        ],
        metadata=metadata,
        # FIXME: Why rely on output_path here, when we use the input path for
        # the zarr_url? That just means that different input & output paths
        # don't work, no?
        image_zarr_path=(Path(output_path) / component),
    )
    original_path_list = parameters["original_paths"]
    num_levels = parameters["num_levels"]
    coarsening_xy = parameters["coarsening_xy"]
    image_extension = parameters["image_extension"]
    image_glob_patterns = parameters["image_glob_patterns"]

    channels: list[OmeroChannel] = get_omero_channel_list(
        image_zarr_path=zarrurl
    )
    wavelength_ids = [c.wavelength_id for c in channels]

    in_path = Path(original_path_list[0])

    # Define well
    component_split = component.split("/")
    well_row = component_split[1]
    well_column = component_split[2]
    well_ID = well_row + well_column

    # Read useful information from ROI table and .zattrs
    adata = read_zarr(f"{zarrurl}/tables/FOV_ROI_table")
    pxl_size = extract_zyx_pixel_sizes(f"{zarrurl}/.zattrs")
    fov_indices = convert_ROI_table_to_indices(
        adata, full_res_pxl_sizes_zyx=pxl_size
    )
    adata_well = read_zarr(f"{zarrurl}/tables/well_ROI_table")
    well_indices = convert_ROI_table_to_indices(
        adata_well, full_res_pxl_sizes_zyx=pxl_size
    )
    if len(well_indices) > 1:
        raise Exception(f"Something wrong with {well_indices=}")

    # FIXME: Put back the choice of columns by name? Not here..

    max_z = well_indices[0][1]
    max_y = well_indices[0][3]
    max_x = well_indices[0][5]

    # Load a single image, to retrieve useful information
    patterns = [f"*_{well_ID}_*.{image_extension}"]
    if image_glob_patterns:
        patterns.extend(image_glob_patterns)
    tmp_images = glob_with_multiple_patterns(
        folder=str(in_path),
        patterns=patterns,
    )
    sample = imread(tmp_images.pop())

    # Initialize zarr
    chunksize = (1, 1, sample.shape[1], sample.shape[2])
    canvas_zarr = zarr.create(
        shape=(len(wavelength_ids), max_z, max_y, max_x),
        chunks=chunksize,
        dtype=sample.dtype,
        store=zarr.storage.FSStore(zarrurl + "/0"),
        overwrite=False,
        dimension_separator="/",
    )

    # Loop over channels
    for i_c, wavelength_id in enumerate(wavelength_ids):
        A, C = wavelength_id.split("_")

        patterns = [f"*_{well_ID}_*{A}*{C}*.{image_extension}"]
        if image_glob_patterns:
            patterns.extend(image_glob_patterns)
        filenames_set = glob_with_multiple_patterns(
            folder=str(in_path),
            patterns=patterns,
        )
        filenames = sorted(list(filenames_set), key=sort_fun)
        if len(filenames) == 0:
            raise Exception(
                "Error in yokogawa_to_ome_zarr: len(filenames)=0.\n"
                f"  in_path: {in_path}\n"
                f"  image_extension: {image_extension}\n"
                f"  well_ID: {well_ID}\n"
                f"  wavelength_id: {wavelength_id},\n"
                f"  patterns: {patterns}"
            )
        # Loop over 3D FOV ROIs
        for indices in fov_indices:
            s_z, e_z, s_y, e_y, s_x, e_x = indices[:]
            region = (
                slice(i_c, i_c + 1),
                slice(s_z, e_z),
                slice(s_y, e_y),
                slice(s_x, e_x),
            )
            FOV_3D = da.concatenate(
                [imread(img) for img in filenames[:e_z]],
            )
            FOV_4D = da.expand_dims(FOV_3D, axis=0)
            filenames = filenames[e_z:]
            da.array(FOV_4D).to_zarr(
                url=canvas_zarr,
                region=region,
                compute=True,
            )

    # Starting from on-disk highest-resolution data, build and write to disk a
    # pyramid of coarser levels
    build_pyramid(
        zarrurl=zarrurl,
        overwrite=False,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        chunksize=chunksize,
    )

    # Deprecated: Delete images (optional)
    # if delete_input:
    #     for f in filenames:
    #         try:
    #             os.remove(f)
    #         except OSError as e:
    #             logging.info("Error: %s : %s" % (f, e.strerror))

    return {}


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=yokogawa_to_ome_zarr,
        logger_name=logger.name,
    )

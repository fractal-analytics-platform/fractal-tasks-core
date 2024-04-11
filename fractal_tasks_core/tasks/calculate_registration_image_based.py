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
Calculates translation for image-based registration
"""
import logging

import anndata as ad
import dask.array as da
import numpy as np
import zarr
from pydantic.decorator import validate_arguments
from skimage.registration import phase_cross_correlation

from fractal_tasks_core.channels import get_channel_from_image_zarr
from fractal_tasks_core.channels import OmeroChannel
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.roi import check_valid_ROI_indices
from fractal_tasks_core.roi import (
    convert_indices_to_regions,
)
from fractal_tasks_core.roi import (
    convert_ROI_table_to_indices,
)
from fractal_tasks_core.roi import load_region
from fractal_tasks_core.tables import write_table
from fractal_tasks_core.tasks._registration_utils import (
    calculate_physical_shifts,
)
from fractal_tasks_core.tasks._registration_utils import (
    get_ROI_table_with_translation,
)
from fractal_tasks_core.tasks.io_models import InitArgsRegistration

logger = logging.getLogger(__name__)


@validate_arguments
def calculate_registration_image_based(
    *,
    # Fractal arguments
    zarr_url: str,
    init_args: InitArgsRegistration,
    # Task-specific arguments
    wavelength_id: str,
    roi_table: str = "FOV_ROI_table",
    level: int = 2,
) -> None:
    """
    Calculate registration based on images

    This task consists of 3 parts:

    1. Loading the images of a given ROI (=> loop over ROIs)
    2. Calculating the transformation for that ROI
    3. Storing the calculated transformation in the ROI table

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        init_args: Intialization arguments provided by
            `image_based_registration_hcs_init`. They contain the
            reference_zarr_url that is used for registration.
            (standard argument for Fractal tasks, managed by Fractal server).
        wavelength_id: Wavelength that will be used for image-based
            registration; e.g. `A01_C01` for Yokogawa, `C01` for MD.
        roi_table: Name of the ROI table over which the task loops to
            calculate the registration. Examples: `FOV_ROI_table` => loop over
            the field of views, `well_ROI_table` => process the whole well as
            one image.
        level: Pyramid level of the image to be used for registration.
            Choose `0` to process at full resolution.

    """
    logger.info(
        f"Running for {zarr_url=}.\n"
        f"Calculating translation registration per {roi_table=} for "
        f"{wavelength_id=}."
    )

    init_args.reference_zarr_url = init_args.reference_zarr_url

    # Read some parameters from Zarr metadata
    ngff_image_meta = load_NgffImageMeta(str(init_args.reference_zarr_url))
    coarsening_xy = ngff_image_meta.coarsening_xy

    # Get channel_index via wavelength_id.
    # Intially only allow registration of the same wavelength
    channel_ref: OmeroChannel = get_channel_from_image_zarr(
        image_zarr_path=init_args.reference_zarr_url,
        wavelength_id=wavelength_id,
    )
    channel_index_ref = channel_ref.index

    channel_align: OmeroChannel = get_channel_from_image_zarr(
        image_zarr_path=zarr_url,
        wavelength_id=wavelength_id,
    )
    channel_index_align = channel_align.index

    # Lazily load zarr array
    data_reference_zyx = da.from_zarr(
        f"{init_args.reference_zarr_url}/{level}"
    )[channel_index_ref]
    data_alignment_zyx = da.from_zarr(f"{zarr_url}/{level}")[
        channel_index_align
    ]

    # Read ROIs
    ROI_table_ref = ad.read_zarr(
        f"{init_args.reference_zarr_url}/tables/{roi_table}"
    )
    ROI_table_x = ad.read_zarr(f"{zarr_url}/tables/{roi_table}")
    logger.info(
        f"Found {len(ROI_table_x)} ROIs in {roi_table=} to be processed."
    )

    # Check that table type of ROI_table_ref is valid. Note that
    # "ngff:region_table" and None are accepted for backwards compatibility
    valid_table_types = [
        "roi_table",
        "masking_roi_table",
        "ngff:region_table",
        None,
    ]
    ROI_table_ref_group = zarr.open_group(
        f"{init_args.reference_zarr_url}/tables/{roi_table}",
        mode="r",
    )
    ref_table_attrs = ROI_table_ref_group.attrs.asdict()
    ref_table_type = ref_table_attrs.get("type")
    if ref_table_type not in valid_table_types:
        raise ValueError(
            (
                f"Table '{roi_table}' (with type '{ref_table_type}') is "
                "not a valid ROI table."
            )
        )

    # For each cycle, get the relevant info
    # TODO: Add additional checks on ROIs?
    if (ROI_table_ref.obs.index != ROI_table_x.obs.index).all():
        raise ValueError(
            "Registration is only implemented for ROIs that match between the "
            "cycles (e.g. well, FOV ROIs). Here, the ROIs in the reference "
            "cycles were {ROI_table_ref.obs.index}, but the ROIs in the "
            "alignment cycle were {ROI_table_x.obs.index}"
        )
    # TODO: Make this less restrictive? i.e. could we also run it if different
    # cycles have different FOVs? But then how do we know which FOVs to match?
    # If we relax this, downstream assumptions on matching based on order
    # in the list will break.

    # Read pixel sizes from zarr attributes
    ngff_image_meta_cycle_x = load_NgffImageMeta(zarr_url)
    pxl_sizes_zyx = ngff_image_meta.get_pixel_sizes_zyx(level=0)
    pxl_sizes_zyx_cycle_x = ngff_image_meta_cycle_x.get_pixel_sizes_zyx(
        level=0
    )

    if pxl_sizes_zyx != pxl_sizes_zyx_cycle_x:
        raise ValueError(
            "Pixel sizes need to be equal between cycles for registration"
        )

    # Create list of indices for 3D ROIs spanning the entire Z direction
    list_indices_ref = convert_ROI_table_to_indices(
        ROI_table_ref,
        level=level,
        coarsening_xy=coarsening_xy,
        full_res_pxl_sizes_zyx=pxl_sizes_zyx,
    )
    check_valid_ROI_indices(list_indices_ref, roi_table)

    list_indices_cycle_x = convert_ROI_table_to_indices(
        ROI_table_x,
        level=level,
        coarsening_xy=coarsening_xy,
        full_res_pxl_sizes_zyx=pxl_sizes_zyx,
    )
    check_valid_ROI_indices(list_indices_cycle_x, roi_table)

    num_ROIs = len(list_indices_ref)
    compute = True
    new_shifts = {}
    for i_ROI in range(num_ROIs):
        logger.info(
            f"Now processing ROI {i_ROI+1}/{num_ROIs} "
            f"for channel {channel_align}."
        )
        img_ref = load_region(
            data_zyx=data_reference_zyx,
            region=convert_indices_to_regions(list_indices_ref[i_ROI]),
            compute=compute,
        )
        img_cycle_x = load_region(
            data_zyx=data_alignment_zyx,
            region=convert_indices_to_regions(list_indices_cycle_x[i_ROI]),
            compute=compute,
        )

        ##############
        #  Calculate the transformation
        ##############
        # Basic version (no padding, no internal binning)
        if img_ref.shape != img_cycle_x.shape:
            raise NotImplementedError(
                "This registration is not implemented for ROIs with "
                "different shapes between cycles"
            )
        shifts = phase_cross_correlation(
            np.squeeze(img_ref), np.squeeze(img_cycle_x)
        )[0]

        # Registration based on scmultiplex, image-based
        # shifts, _, _ = calculate_shift(np.squeeze(img_ref),
        #           np.squeeze(img_cycle_x), bin=binning, binarize=False)

        # TODO: Make this work on label images
        # (=> different loading) etc.

        ##############
        # Storing the calculated transformation ###
        ##############
        # Store the shift in ROI table
        # TODO: Store in OME-NGFF transformations: Check SpatialData approach,
        # per ROI storage?

        # Adapt ROIs for the given ROI table:
        ROI_name = ROI_table_ref.obs.index[i_ROI]
        new_shifts[ROI_name] = calculate_physical_shifts(
            shifts,
            level=level,
            coarsening_xy=coarsening_xy,
            full_res_pxl_sizes_zyx=pxl_sizes_zyx,
        )

    # Write physical shifts to disk (as part of the ROI table)
    logger.info(f"Updating the {roi_table=} with translation columns")
    image_group = zarr.group(zarr_url)
    new_ROI_table = get_ROI_table_with_translation(ROI_table_x, new_shifts)
    write_table(
        image_group,
        roi_table,
        new_ROI_table,
        overwrite=True,
        table_attrs=ref_table_attrs,
    )


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=calculate_registration_image_based,
        logger_name=logger.name,
    )

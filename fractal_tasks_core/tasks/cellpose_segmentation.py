# Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Tommaso Comparin <tommaso.comparin@exact-lab.it>
# Marco Franzon <marco.franzon@exact-lab.it>
# Joel Lüthi  <joel.luethi@fmi.ch>
#
# This file is part of Fractal and was originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
"""
Image segmentation via Cellpose library.
"""
import logging
import os
import time
from typing import Literal
from typing import Optional

import anndata as ad
import cellpose
import dask.array as da
import numpy as np
import zarr
from cellpose import models
from pydantic import Field
from pydantic import validate_call

import fractal_tasks_core
from fractal_tasks_core.labels import prepare_label_group
from fractal_tasks_core.masked_loading import masked_loading_wrapper
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.pyramids import build_pyramid
from fractal_tasks_core.roi import (
    array_to_bounding_box_table,
)
from fractal_tasks_core.roi import check_valid_ROI_indices
from fractal_tasks_core.roi import (
    convert_ROI_table_to_indices,
)
from fractal_tasks_core.roi import create_roi_table_from_df_list
from fractal_tasks_core.roi import (
    find_overlaps_in_ROI_indices,
)
from fractal_tasks_core.roi import get_overlapping_pairs_3D
from fractal_tasks_core.roi import is_ROI_table_valid
from fractal_tasks_core.roi import load_region
from fractal_tasks_core.tables import write_table
from fractal_tasks_core.tasks.cellpose_utils import (
    _normalize_cellpose_channels,
)
from fractal_tasks_core.tasks.cellpose_utils import (
    CellposeChannel1InputModel,
)
from fractal_tasks_core.tasks.cellpose_utils import (
    CellposeChannel2InputModel,
)
from fractal_tasks_core.tasks.cellpose_utils import (
    CellposeCustomNormalizer,
)
from fractal_tasks_core.tasks.cellpose_utils import (
    CellposeModelParams,
)
from fractal_tasks_core.utils import rescale_datasets

logger = logging.getLogger(__name__)

__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__


def segment_ROI(
    x: np.ndarray,
    num_labels_tot: dict[str, int],
    model: models.CellposeModel = None,
    do_3D: bool = True,
    channels: list[int] = [0, 0],
    diameter: float = 30.0,
    normalize: CellposeCustomNormalizer = CellposeCustomNormalizer(),
    normalize2: Optional[CellposeCustomNormalizer] = None,
    label_dtype: Optional[np.dtype] = None,
    relabeling: bool = True,
    advanced_cellpose_model_params: CellposeModelParams = CellposeModelParams(),  # noqa: E501
) -> np.ndarray:
    """
    Internal function that runs Cellpose segmentation for a single ROI.

    Args:
        x: 4D numpy array.
        num_labels_tot: Number of labels already in total image. Used for
            relabeling purposes. Using a dict to have a mutable object that
            can be edited from within the function without having to be passed
            back through the masked_loading_wrapper.
        model: An instance of `models.CellposeModel`.
        do_3D: If `True`, cellpose runs in 3D mode: runs on xy, xz & yz planes,
            then averages the flows.
        channels: Which channels to use. If only one channel is provided, `[0,
            0]` should be used. If two channels are provided (the first
            dimension of `x` has length of 2), `[1, 2]` should be used
            (`x[0, :, :,:]` contains the membrane channel and
            `x[1, :, :, :]` contains the nuclear channel).
        diameter: Expected object diameter in pixels for cellpose.
        normalize: By default, data is normalized so 0.0=1st percentile and
            1.0=99th percentile of image intensities in each channel.
            This automatic normalization can lead to issues when the image to
            be segmented is very sparse. You can turn off the default
            rescaling. With the "custom" option, you can either provide your
            own rescaling percentiles or fixed rescaling upper and lower
            bound integers.
        normalize2: Normalization options for channel 2. If one channel is
            normalized with default settings, both channels need to be
            normalized with default settings.
        label_dtype: Label images are cast into this `np.dtype`.
        relabeling: Whether relabeling based on num_labels_tot is performed.
        advanced_cellpose_model_params: Advanced Cellpose model parameters
            that are passed to the Cellpose `model.eval` method.
    """

    # Write some debugging info
    logger.info(
        "[segment_ROI] START |"
        f" x: {type(x)}, {x.shape} |"
        f" {do_3D=} |"
        f" {model.diam_mean=} |"
        f" {diameter=} |"
        f" {advanced_cellpose_model_params.flow_threshold=} |"
        f" {normalize.type=}"
    )
    x = _normalize_cellpose_channels(x, channels, normalize, normalize2)

    # Actual labeling
    t0 = time.perf_counter()
    mask, _, _ = model.eval(
        x,
        channels=channels,
        do_3D=do_3D,
        net_avg=advanced_cellpose_model_params.net_avg,
        augment=advanced_cellpose_model_params.augment,
        diameter=diameter,
        anisotropy=advanced_cellpose_model_params.anisotropy,
        cellprob_threshold=advanced_cellpose_model_params.cellprob_threshold,
        flow_threshold=advanced_cellpose_model_params.flow_threshold,
        normalize=normalize.cellpose_normalize,
        min_size=advanced_cellpose_model_params.min_size,
        batch_size=advanced_cellpose_model_params.batch_size,
        invert=advanced_cellpose_model_params.invert,
        tile=advanced_cellpose_model_params.tile,
        tile_overlap=advanced_cellpose_model_params.tile_overlap,
        resample=advanced_cellpose_model_params.resample,
        interp=advanced_cellpose_model_params.interp,
        stitch_threshold=advanced_cellpose_model_params.stitch_threshold,
    )

    if mask.ndim == 2:
        # If we get a 2D image, we still return it as a 3D array
        mask = np.expand_dims(mask, axis=0)
    t1 = time.perf_counter()

    # Write some debugging info
    logger.info(
        "[segment_ROI] END   |"
        f" Elapsed: {t1-t0:.3f} s |"
        f" {mask.shape=},"
        f" {mask.dtype=} (then {label_dtype}),"
        f" {np.max(mask)=} |"
        f" {model.diam_mean=} |"
        f" {diameter=} |"
        f" {advanced_cellpose_model_params.flow_threshold=}"
    )

    # Shift labels and update relabeling counters
    if relabeling:
        num_labels_roi = np.max(mask)
        mask[mask > 0] += num_labels_tot["num_labels_tot"]
        num_labels_tot["num_labels_tot"] += num_labels_roi

        # Write some logs
        logger.info(f"ROI had {num_labels_roi=}, {num_labels_tot=}")

        # Check that total number of labels is under control
        if num_labels_tot["num_labels_tot"] > np.iinfo(label_dtype).max:
            raise ValueError(
                "ERROR in re-labeling:"
                f"Reached {num_labels_tot} labels, "
                f"but dtype={label_dtype}"
            )

    return mask.astype(label_dtype)


@validate_call
def cellpose_segmentation(
    *,
    # Fractal parameters
    zarr_url: str,
    # Core parameters
    level: int,
    channel: CellposeChannel1InputModel,
    channel2: CellposeChannel2InputModel = Field(
        default_factory=CellposeChannel2InputModel
    ),
    input_ROI_table: str = "FOV_ROI_table",
    output_ROI_table: Optional[str] = None,
    output_label_name: Optional[str] = None,
    # Cellpose-related arguments
    diameter_level0: float = 30.0,
    # https://github.com/fractal-analytics-platform/fractal-tasks-core/issues/401 # noqa E501
    model_type: Literal[tuple(models.MODEL_NAMES)] = "cyto2",  # type: ignore
    pretrained_model: Optional[str] = None,
    relabeling: bool = True,
    use_masks: bool = True,
    advanced_cellpose_model_params: CellposeModelParams = Field(
        default_factory=CellposeModelParams
    ),
    overwrite: bool = True,
) -> None:
    """
    Run cellpose segmentation on the ROIs of a single OME-Zarr image.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        level: Pyramid level of the image to be segmented. Choose `0` to
            process at full resolution.
        channel: Primary channel for segmentation; requires either
            `wavelength_id` (e.g. `A01_C01`) or `label` (e.g. `DAPI`), but not
            both. Also contains normalization options. By default, data is
            normalized so 0.0=1st percentile and 1.0=99th percentile of image
            intensities in each channel.
            This automatic normalization can lead to issues when the image to
            be segmented is very sparse. You can turn off the default
            rescaling. With the "custom" option, you can either provide your
            own rescaling percentiles or fixed rescaling upper and lower
            bound integers.
        channel2: Second channel for segmentation (in the same format as
            `channel`). If specified, cellpose runs in dual channel mode.
            For dual channel segmentation of cells, the first channel should
            contain the membrane marker, the second channel should contain the
            nuclear marker.
        input_ROI_table: Name of the ROI table over which the task loops to
            apply Cellpose segmentation. Examples: `FOV_ROI_table` => loop over
            the field of views, `organoid_ROI_table` => loop over the organoid
            ROI table (generated by another task), `well_ROI_table` => process
            the whole well as one image.
        output_ROI_table: If provided, a ROI table with that name is created,
            which will contain the bounding boxes of the newly segmented
            labels. ROI tables should have `ROI` in their name.
        output_label_name: Name of the output label image (e.g. `"organoids"`).
        diameter_level0: Expected diameter of the objects that should be
            segmented in pixels at level 0. Initial diameter is rescaled using
            the `level` that was selected. The rescaled value is passed as
            the diameter to the `CellposeModel.eval` method.
        model_type: Parameter of `CellposeModel` class. Defines which model
            should be used. Typical choices are `nuclei`, `cyto`, `cyto2`, etc.
        pretrained_model: Parameter of `CellposeModel` class (takes
            precedence over `model_type`). Allows you to specify the path of
            a custom trained cellpose model.
        relabeling: If `True`, apply relabeling so that label values are
            unique for all objects in the well.
        use_masks: If `True`, try to use masked loading and fall back to
            `use_masks=False` if the ROI table is not suitable. Masked
            loading is relevant when only a subset of the bounding box should
            actually be processed (e.g. running within `organoid_ROI_table`).
        advanced_cellpose_model_params: Advanced Cellpose model parameters
            that are passed to the Cellpose `model.eval` method.
        overwrite: If `True`, overwrite the task output.
    """
    logger.info(f"Processing {zarr_url=}")

    # Preliminary checks on Cellpose model
    if pretrained_model:
        if not os.path.exists(pretrained_model):
            raise ValueError(f"{pretrained_model=} does not exist.")

    # Read attributes from NGFF metadata
    ngff_image_meta = load_NgffImageMeta(zarr_url)
    num_levels = ngff_image_meta.num_levels
    coarsening_xy = ngff_image_meta.coarsening_xy
    full_res_pxl_sizes_zyx = ngff_image_meta.get_pixel_sizes_zyx(level=0)
    actual_res_pxl_sizes_zyx = ngff_image_meta.get_pixel_sizes_zyx(level=level)
    logger.info(f"NGFF image has {num_levels=}")
    logger.info(f"NGFF image has {coarsening_xy=}")
    logger.info(
        f"NGFF image has full-res pixel sizes {full_res_pxl_sizes_zyx}"
    )
    logger.info(
        f"NGFF image has level-{level} pixel sizes "
        f"{actual_res_pxl_sizes_zyx}"
    )

    # Find channel index
    omero_channel = channel.get_omero_channel(zarr_url)
    if omero_channel:
        ind_channel = omero_channel.index
    else:
        return

    # Find channel index for second channel, if one is provided
    if channel2.is_set():
        omero_channel_2 = channel2.get_omero_channel(zarr_url)
        if omero_channel_2:
            ind_channel_c2 = omero_channel_2.index
        else:
            return

    # Set channel label
    if output_label_name is None:
        try:
            channel_label = omero_channel.label
            output_label_name = f"label_{channel_label}"
        except (KeyError, IndexError):
            output_label_name = f"label_{ind_channel}"

    # Load ZYX data
    # Workaround for #788: Only load channel index when there is a channel
    # dimension
    if ngff_image_meta.axes_names[0] != "c":
        data_zyx = da.from_zarr(f"{zarr_url}/{level}")
        if channel2.is_set():
            raise ValueError(
                "Dual channel input was specified for an OME-Zarr image "
                "without a channel axis"
            )
    else:
        data_zyx = da.from_zarr(f"{zarr_url}/{level}")[ind_channel]
        if channel2.is_set():
            data_zyx_c2 = da.from_zarr(f"{zarr_url}/{level}")[ind_channel_c2]
            logger.info(f"Second channel: {data_zyx_c2.shape=}")
    logger.info(f"{data_zyx.shape=}")

    # Read ROI table
    ROI_table_path = f"{zarr_url}/tables/{input_ROI_table}"
    ROI_table = ad.read_zarr(ROI_table_path)

    # Perform some checks on the ROI table
    valid_ROI_table = is_ROI_table_valid(
        table_path=ROI_table_path, use_masks=use_masks
    )
    if use_masks and not valid_ROI_table:
        logger.info(
            f"ROI table at {ROI_table_path} cannot be used for masked "
            "loading. Set use_masks=False."
        )
        use_masks = False
    logger.info(f"{use_masks=}")

    # Create list of indices for 3D ROIs spanning the entire Z direction
    list_indices = convert_ROI_table_to_indices(
        ROI_table,
        level=level,
        coarsening_xy=coarsening_xy,
        full_res_pxl_sizes_zyx=full_res_pxl_sizes_zyx,
    )
    check_valid_ROI_indices(list_indices, input_ROI_table)

    # If we are not planning to use masked loading, fail for overlapping ROIs
    if not use_masks:
        overlap = find_overlaps_in_ROI_indices(list_indices)
        if overlap:
            raise ValueError(
                f"ROI indices created from {input_ROI_table} table have "
                "overlaps, but we are not using masked loading."
            )

    # Select 2D/3D behavior and set some parameters
    do_3D = data_zyx.shape[0] > 1 and len(data_zyx.shape) == 3
    if do_3D:
        if advanced_cellpose_model_params.anisotropy is None:
            # Compute anisotropy as pixel_size_z/pixel_size_x
            advanced_cellpose_model_params.anisotropy = (
                actual_res_pxl_sizes_zyx[0] / actual_res_pxl_sizes_zyx[2]
            )
        logger.info(f"Anisotropy: {advanced_cellpose_model_params.anisotropy}")

    # Rescale datasets (only relevant for level>0)
    # Workaround for #788
    if ngff_image_meta.axes_names[0] != "c":
        new_datasets = rescale_datasets(
            datasets=[ds.model_dump() for ds in ngff_image_meta.datasets],
            coarsening_xy=coarsening_xy,
            reference_level=level,
            remove_channel_axis=False,
        )
    else:
        new_datasets = rescale_datasets(
            datasets=[ds.model_dump() for ds in ngff_image_meta.datasets],
            coarsening_xy=coarsening_xy,
            reference_level=level,
            remove_channel_axis=True,
        )

    label_attrs = {
        "image-label": {
            "version": __OME_NGFF_VERSION__,
            "source": {"image": "../../"},
        },
        "multiscales": [
            {
                "name": output_label_name,
                "version": __OME_NGFF_VERSION__,
                "axes": [
                    ax.dict()
                    for ax in ngff_image_meta.multiscale.axes
                    if ax.type != "channel"
                ],
                "datasets": new_datasets,
            }
        ],
    }

    image_group = zarr.group(zarr_url)
    label_group = prepare_label_group(
        image_group,
        output_label_name,
        overwrite=overwrite,
        label_attrs=label_attrs,
        logger=logger,
    )

    logger.info(
        f"Helper function `prepare_label_group` returned {label_group=}"
    )
    logger.info(f"Output label path: {zarr_url}/labels/{output_label_name}/0")
    store = zarr.storage.FSStore(f"{zarr_url}/labels/{output_label_name}/0")
    label_dtype = np.uint32

    # Ensure that all output shapes & chunks are 3D (for 2D data: (1, y, x))
    # https://github.com/fractal-analytics-platform/fractal-tasks-core/issues/398
    shape = data_zyx.shape
    if len(shape) == 2:
        shape = (1, *shape)
    chunks = data_zyx.chunksize
    if len(chunks) == 2:
        chunks = (1, *chunks)
    mask_zarr = zarr.create(
        shape=shape,
        chunks=chunks,
        dtype=label_dtype,
        store=store,
        overwrite=False,
        dimension_separator="/",
    )

    logger.info(
        f"mask will have shape {data_zyx.shape} "
        f"and chunks {data_zyx.chunks}"
    )

    # Initialize cellpose
    gpu = advanced_cellpose_model_params.use_gpu and cellpose.core.use_gpu()
    if pretrained_model:
        model = models.CellposeModel(
            gpu=gpu, pretrained_model=pretrained_model
        )
    else:
        model = models.CellposeModel(gpu=gpu, model_type=model_type)

    # Initialize other things
    logger.info(f"Start cellpose_segmentation task for {zarr_url}")
    logger.info(f"relabeling: {relabeling}")
    logger.info(f"do_3D: {do_3D}")
    logger.info(f"use_gpu: {gpu}")
    logger.info(f"level: {level}")
    logger.info(f"model_type: {model_type}")
    logger.info(f"pretrained_model: {pretrained_model}")
    logger.info(f"anisotropy: {advanced_cellpose_model_params.anisotropy}")
    logger.info("Total well shape/chunks:")
    logger.info(f"{data_zyx.shape}")
    logger.info(f"{data_zyx.chunks}")
    if channel2.is_set():
        logger.info("Dual channel input for cellpose model")
        logger.info(f"{data_zyx_c2.shape}")
        logger.info(f"{data_zyx_c2.chunks}")

    # Counters for relabeling
    num_labels_tot = {"num_labels_tot": 0}

    # Iterate over ROIs
    num_ROIs = len(list_indices)

    if output_ROI_table:
        bbox_dataframe_list = []

    logger.info(f"Now starting loop over {num_ROIs} ROIs")
    for i_ROI, indices in enumerate(list_indices):
        # Define region
        s_z, e_z, s_y, e_y, s_x, e_x = indices[:]
        region = (
            slice(s_z, e_z),
            slice(s_y, e_y),
            slice(s_x, e_x),
        )
        logger.info(f"Now processing ROI {i_ROI+1}/{num_ROIs}")

        # Prepare single-channel or dual-channel input for cellpose
        if channel2.is_set():
            # Dual channel mode, first channel is the membrane channel
            img_1 = load_region(
                data_zyx,
                region,
                compute=True,
                return_as_3D=True,
            )
            img_np = np.zeros((2, *img_1.shape))
            img_np[0, :, :, :] = img_1
            img_np[1, :, :, :] = load_region(
                data_zyx_c2,
                region,
                compute=True,
                return_as_3D=True,
            )
            channels = [1, 2]
        else:
            img_np = np.expand_dims(
                load_region(data_zyx, region, compute=True, return_as_3D=True),
                axis=0,
            )
            channels = [0, 0]

        # Prepare keyword arguments for segment_ROI function
        kwargs_segment_ROI = dict(
            num_labels_tot=num_labels_tot,
            model=model,
            channels=channels,
            do_3D=do_3D,
            label_dtype=label_dtype,
            diameter=diameter_level0 / coarsening_xy**level,
            normalize=channel.normalize,
            normalize2=channel2.normalize,
            relabeling=relabeling,
            advanced_cellpose_model_params=advanced_cellpose_model_params,
        )

        # Prepare keyword arguments for preprocessing function
        preprocessing_kwargs = {}
        if use_masks:
            preprocessing_kwargs = dict(
                region=region,
                current_label_path=f"{zarr_url}/labels/{output_label_name}/0",
                ROI_table_path=ROI_table_path,
                ROI_positional_index=i_ROI,
            )

        # Call segment_ROI through the masked-loading wrapper, which includes
        # pre/post-processing functions if needed
        new_label_img = masked_loading_wrapper(
            image_array=img_np,
            function=segment_ROI,
            kwargs=kwargs_segment_ROI,
            use_masks=use_masks,
            preprocessing_kwargs=preprocessing_kwargs,
        )

        if output_ROI_table:
            bbox_df = array_to_bounding_box_table(
                new_label_img,
                actual_res_pxl_sizes_zyx,
                origin_zyx=(s_z, s_y, s_x),
            )

            bbox_dataframe_list.append(bbox_df)

            overlap_list = get_overlapping_pairs_3D(
                bbox_df, full_res_pxl_sizes_zyx
            )
            if len(overlap_list) > 0:
                logger.warning(
                    f"ROI {indices} has "
                    f"{len(overlap_list)} bounding-box pairs overlap"
                )

        # Compute and store 0-th level to disk
        da.array(new_label_img).to_zarr(
            url=mask_zarr,
            region=region,
            compute=True,
        )

    logger.info(
        f"End cellpose_segmentation task for {zarr_url}, "
        "now building pyramids."
    )

    # Starting from on-disk highest-resolution data, build and write to disk a
    # pyramid of coarser levels
    build_pyramid(
        zarrurl=f"{zarr_url}/labels/{output_label_name}",
        overwrite=overwrite,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        chunksize=chunks,
        aggregation_function=np.max,
    )

    logger.info("End building pyramids")

    if output_ROI_table:
        bbox_table = create_roi_table_from_df_list(bbox_dataframe_list)

        # Write to zarr group
        image_group = zarr.group(zarr_url)
        logger.info(
            "Now writing bounding-box ROI table to "
            f"{zarr_url}/tables/{output_ROI_table}"
        )
        table_attrs = {
            "type": "masking_roi_table",
            "region": {"path": f"../labels/{output_label_name}"},
            "instance_key": "label",
        }
        write_table(
            image_group,
            output_ROI_table,
            bbox_table,
            overwrite=overwrite,
            table_attrs=table_attrs,
        )


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=cellpose_segmentation,
        logger_name=logger.name,
    )

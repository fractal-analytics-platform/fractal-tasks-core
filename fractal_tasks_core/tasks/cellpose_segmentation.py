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
from pathlib import Path
from typing import Any
from typing import Literal
from typing import Optional
from typing import Sequence

import anndata as ad
import cellpose
import dask.array as da
import numpy as np
import pandas as pd
import zarr
from cellpose import models
from pydantic.decorator import validate_arguments

import fractal_tasks_core
from fractal_tasks_core.channels import ChannelInputModel
from fractal_tasks_core.channels import ChannelNotFoundError
from fractal_tasks_core.channels import get_channel_from_image_zarr
from fractal_tasks_core.channels import OmeroChannel
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
from fractal_tasks_core.roi import empty_bounding_box_table
from fractal_tasks_core.roi import (
    find_overlaps_in_ROI_indices,
)
from fractal_tasks_core.roi import get_overlapping_pairs_3D
from fractal_tasks_core.roi import is_ROI_table_valid
from fractal_tasks_core.roi import load_region
from fractal_tasks_core.tables import write_table
from fractal_tasks_core.tasks.cellpose_transforms import (
    CellposeCustomNormalizer,
)
from fractal_tasks_core.tasks.cellpose_transforms import normalized_img
from fractal_tasks_core.utils import rescale_datasets

logger = logging.getLogger(__name__)

__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__


def segment_ROI(
    x: np.ndarray,
    model: models.CellposeModel = None,
    do_3D: bool = True,
    channels: list[int] = [0, 0],
    anisotropy: Optional[float] = None,
    diameter: float = 30.0,
    cellprob_threshold: float = 0.0,
    flow_threshold: float = 0.4,
    normalize: CellposeCustomNormalizer = CellposeCustomNormalizer(),
    label_dtype: Optional[np.dtype] = None,
    augment: bool = False,
    net_avg: bool = False,
    min_size: int = 15,
    batch_size: int = 8,
    invert: bool = False,
    tile: bool = True,
    tile_overlap: float = 0.1,
    resample: bool = True,
    interp: bool = True,
    stitch_threshold: float = 0.0,
) -> np.ndarray:
    """
    Internal function that runs Cellpose segmentation for a single ROI.

    Args:
        x: 4D numpy array.
        model: An instance of `models.CellposeModel`.
        do_3D: If `True`, cellpose runs in 3D mode: runs on xy, xz & yz planes,
            then averages the flows.
        channels: Which channels to use. If only one channel is provided, `[0,
            0]` should be used. If two channels are provided (the first
            dimension of `x` has length of 2), `[1, 2]` should be used
            (`x[0, :, :,:]` contains the membrane channel and
            `x[1, :, :, :]` contains the nuclear channel).
        anisotropy: Set anisotropy rescaling factor for Z dimension.
        diameter: Expected object diameter in pixels for cellpose.
        cellprob_threshold: Cellpose model parameter.
        flow_threshold: Cellpose model parameter.
        normalize: normalize data so 0.0=1st percentile and 1.0=99th
            percentile of image intensities in each channel. This automatic
            normalization can lead to issues when the image to be segmented
            is very sparse.
        label_dtype: Label images are cast into this `np.dtype`.
        augment: Whether to use cellpose augmentation to tile images with
            overlap.
        net_avg: Whether to use cellpose net averaging to run the 4 built-in
            networks (useful for `nuclei`, `cyto` and `cyto2`, not sure it
            works for the others).
        min_size: Minimum size of the segmented objects.
        batch_size: number of 224x224 patches to run simultaneously on the GPU
            (can make smaller or bigger depending on GPU memory usage)
        invert: invert image pixel intensity before running network (if True,
            image is also normalized)
        tile: tiles image to ensure GPU/CPU memory usage limited (recommended)
        tile_overlap: fraction of overlap of tiles when computing flows
        resample: run dynamics at original image size (will be slower but
            create more accurate boundaries)
        interp: interpolate during 2D dynamics (not available in 3D)
            (in previous versions it was False, now it defaults to True)
        stitch_threshold: if stitch_threshold>0.0 and not do_3D and equal
            image sizes, masks are stitched in 3D to return volume segmentation
    """

    # Write some debugging info
    logger.info(
        "[segment_ROI] START |"
        f" x: {type(x)}, {x.shape} |"
        f" {do_3D=} |"
        f" {model.diam_mean=} |"
        f" {diameter=} |"
        f" {flow_threshold=} |"
        f" {normalize.type=}"
    )

    # Optionally perform custom normalization
    if normalize.type == "custom":
        x = normalized_img(
            x,
            lower_p=normalize.lower_percentile,
            upper_p=normalize.upper_percentile,
            lower_bound=normalize.lower_bound,
            upper_bound=normalize.upper_bound,
        )

    # Actual labeling
    t0 = time.perf_counter()
    mask, _, _ = model.eval(
        x,
        channels=channels,
        do_3D=do_3D,
        net_avg=net_avg,
        augment=augment,
        diameter=diameter,
        anisotropy=anisotropy,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
        normalize=normalize.cellpose_normalize,
        min_size=min_size,
        batch_size=batch_size,
        invert=invert,
        tile=tile,
        tile_overlap=tile_overlap,
        resample=resample,
        interp=interp,
        stitch_threshold=stitch_threshold,
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
        f" {flow_threshold=}"
    )

    return mask.astype(label_dtype)


@validate_arguments
def cellpose_segmentation(
    *,
    # Fractal arguments
    input_paths: Sequence[str],
    output_path: str,
    component: str,
    metadata: dict[str, Any],
    # Task-specific arguments
    level: int,
    channel: ChannelInputModel,
    channel2: Optional[ChannelInputModel] = None,
    input_ROI_table: str = "FOV_ROI_table",
    output_ROI_table: Optional[str] = None,
    output_label_name: Optional[str] = None,
    use_masks: bool = True,
    relabeling: bool = True,
    # Cellpose-related arguments
    diameter_level0: float = 30.0,
    # https://github.com/fractal-analytics-platform/fractal-tasks-core/issues/401 # noqa E501
    model_type: Literal[tuple(models.MODEL_NAMES)] = "cyto2",
    pretrained_model: Optional[str] = None,
    cellprob_threshold: float = 0.0,
    flow_threshold: float = 0.4,
    normalize: CellposeCustomNormalizer = CellposeCustomNormalizer(),
    anisotropy: Optional[float] = None,
    min_size: int = 15,
    augment: bool = False,
    net_avg: bool = False,
    use_gpu: bool = True,
    batch_size: int = 8,
    invert: bool = False,
    tile: bool = True,
    tile_overlap: float = 0.1,
    resample: bool = True,
    interp: bool = True,
    stitch_threshold: float = 0.0,
    # Overwrite option
    overwrite: bool = True,
) -> dict[str, Any]:
    """
    Run cellpose segmentation on the ROIs of a single OME-Zarr image.

    Args:
        input_paths: List of input paths where the image data is stored as
            OME-Zarrs. Should point to the parent folder containing one or many
            OME-Zarr files, not the actual OME-Zarr file. Example:
            `["/some/path/"]`. This task only supports a single input path.
            (standard argument for Fractal tasks, managed by Fractal server).
        output_path: This parameter is not used by this task.
            (standard argument for Fractal tasks, managed by Fractal server).
        component: Path to the OME-Zarr image in the OME-Zarr plate that is
            processed. Example: `"some_plate.zarr/B/03/0"`.
            (standard argument for Fractal tasks, managed by Fractal server).
        metadata: This parameter is not used by this task.
            (standard argument for Fractal tasks, managed by Fractal server).
        level: Pyramid level of the image to be segmented. Choose `0` to
            process at full resolution.
        channel: Primary channel for segmentation; requires either
            `wavelength_id` (e.g. `A01_C01`) or `label` (e.g. `DAPI`).
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
        use_masks: If `True`, try to use masked loading and fall back to
            `use_masks=False` if the ROI table is not suitable. Masked
            loading is relevant when only a subset of the bounding box should
            actually be processed (e.g. running within `organoid_ROI_table`).
        output_label_name: Name of the output label image (e.g. `"organoids"`).
        relabeling: If `True`, apply relabeling so that label values are
            unique for all objects in the well.
        diameter_level0: Expected diameter of the objects that should be
            segmented in pixels at level 0. Initial diameter is rescaled using
            the `level` that was selected. The rescaled value is passed as
            the diameter to the `CellposeModel.eval` method.
        model_type: Parameter of `CellposeModel` class. Defines which model
            should be used. Typical choices are `nuclei`, `cyto`, `cyto2`, etc.
        pretrained_model: Parameter of `CellposeModel` class (takes
            precedence over `model_type`). Allows you to specify the path of
            a custom trained cellpose model.
        cellprob_threshold: Parameter of `CellposeModel.eval` method. Valid
            values between -6 to 6. From Cellpose documentation: "Decrease this
            threshold if cellpose is not returning as many ROIs as you’d
            expect. Similarly, increase this threshold if cellpose is returning
            too ROIs particularly from dim areas."
        flow_threshold: Parameter of `CellposeModel.eval` method. Valid
            values between 0.0 and 1.0. From Cellpose documentation: "Increase
            this threshold if cellpose is not returning as many ROIs as you’d
            expect. Similarly, decrease this threshold if cellpose is returning
            too many ill-shaped ROIs."
        normalize: By default, data is normalized so 0.0=1st percentile and
            1.0=99th percentile of image intensities in each channel.
            This automatic normalization can lead to issues when the image to
            be segmented is very sparse. You can turn off the default
            rescaling. With the "custom" option, you can either provide your
            own rescaling percentiles or fixed rescaling upper and lower
            bound integers.
        anisotropy: Ratio of the pixel sizes along Z and XY axis (ignored if
            the image is not three-dimensional). If `None`, it is inferred from
            the OME-NGFF metadata.
        min_size: Parameter of `CellposeModel` class. Minimum size of the
            segmented objects (in pixels). Use `-1` to turn off the size
            filter.
        augment: Parameter of `CellposeModel` class. Whether to use cellpose
            augmentation to tile images with overlap.
        net_avg: Parameter of `CellposeModel` class. Whether to use cellpose
            net averaging to run the 4 built-in networks (useful for `nuclei`,
            `cyto` and `cyto2`, not sure it works for the others).
        use_gpu: If `False`, always use the CPU; if `True`, use the GPU if
            possible (as defined in `cellpose.core.use_gpu()`) and fall-back
            to the CPU otherwise.
        batch_size: number of 224x224 patches to run simultaneously on the GPU
            (can make smaller or bigger depending on GPU memory usage)
        invert: invert image pixel intensity before running network (if True,
            image is also normalized)
        tile: tiles image to ensure GPU/CPU memory usage limited (recommended)
        tile_overlap: fraction of overlap of tiles when computing flows
        resample: run dynamics at original image size (will be slower but
            create more accurate boundaries)
        interp: interpolate during 2D dynamics (not available in 3D)
            (in previous versions it was False, now it defaults to True)
        stitch_threshold: if stitch_threshold>0.0 and not do_3D and equal
            image sizes, masks are stitched in 3D to return volume segmentation
        overwrite: If `True`, overwrite the task output.
    """

    # Set input path
    if len(input_paths) > 1:
        raise NotImplementedError
    in_path = Path(input_paths[0])
    zarrurl = (in_path.resolve() / component).as_posix()
    logger.info(f"{zarrurl=}")

    # Preliminary checks on Cellpose model
    if pretrained_model:
        if not os.path.exists(pretrained_model):
            raise ValueError(f"{pretrained_model=} does not exist.")

    # Read attributes from NGFF metadata
    ngff_image_meta = load_NgffImageMeta(zarrurl)
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
    try:
        tmp_channel: OmeroChannel = get_channel_from_image_zarr(
            image_zarr_path=zarrurl,
            wavelength_id=channel.wavelength_id,
            label=channel.label,
        )
    except ChannelNotFoundError as e:
        logger.warning(
            "Channel not found, exit from the task.\n"
            f"Original error: {str(e)}"
        )
        return {}
    ind_channel = tmp_channel.index

    # Find channel index for second channel, if one is provided
    if channel2:
        try:
            tmp_channel_c2: OmeroChannel = get_channel_from_image_zarr(
                image_zarr_path=zarrurl,
                wavelength_id=channel2.wavelength_id,
                label=channel2.label,
            )
        except ChannelNotFoundError as e:
            logger.warning(
                f"Second channel with wavelength_id: {channel2.wavelength_id} "
                f"and label: {channel2.label} not found, exit from the task.\n"
                f"Original error: {str(e)}"
            )
            return {}
        ind_channel_c2 = tmp_channel_c2.index

    # Set channel label
    if output_label_name is None:
        try:
            channel_label = tmp_channel.label
            output_label_name = f"label_{channel_label}"
        except (KeyError, IndexError):
            output_label_name = f"label_{ind_channel}"

    # Load ZYX data
    data_zyx = da.from_zarr(f"{zarrurl}/{level}")[ind_channel]
    logger.info(f"{data_zyx.shape=}")
    if channel2:
        data_zyx_c2 = da.from_zarr(f"{zarrurl}/{level}")[ind_channel_c2]
        logger.info(f"Second channel: {data_zyx_c2.shape=}")

    # Read ROI table
    ROI_table_path = f"{zarrurl}/tables/{input_ROI_table}"
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
        if anisotropy is None:
            # Compute anisotropy as pixel_size_z/pixel_size_x
            anisotropy = (
                actual_res_pxl_sizes_zyx[0] / actual_res_pxl_sizes_zyx[2]
            )
        logger.info(f"Anisotropy: {anisotropy}")

    # Rescale datasets (only relevant for level>0)
    if ngff_image_meta.axes_names[0] != "c":
        raise ValueError(
            "Cannot set `remove_channel_axis=True` for multiscale "
            f"metadata with axes={ngff_image_meta.axes_names}. "
            'First axis should have name "c".'
        )
    new_datasets = rescale_datasets(
        datasets=[ds.dict() for ds in ngff_image_meta.datasets],
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

    image_group = zarr.group(zarrurl)
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
    logger.info(f"Output label path: {zarrurl}/labels/{output_label_name}/0")
    store = zarr.storage.FSStore(f"{zarrurl}/labels/{output_label_name}/0")
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
    gpu = use_gpu and cellpose.core.use_gpu()
    if pretrained_model:
        model = models.CellposeModel(
            gpu=gpu, pretrained_model=pretrained_model
        )
    else:
        model = models.CellposeModel(gpu=gpu, model_type=model_type)

    # Initialize other things
    logger.info(f"Start cellpose_segmentation task for {zarrurl}")
    logger.info(f"relabeling: {relabeling}")
    logger.info(f"do_3D: {do_3D}")
    logger.info(f"use_gpu: {gpu}")
    logger.info(f"level: {level}")
    logger.info(f"model_type: {model_type}")
    logger.info(f"pretrained_model: {pretrained_model}")
    logger.info(f"anisotropy: {anisotropy}")
    logger.info("Total well shape/chunks:")
    logger.info(f"{data_zyx.shape}")
    logger.info(f"{data_zyx.chunks}")
    if channel2:
        logger.info("Dual channel input for cellpose model")
        logger.info(f"{data_zyx_c2.shape}")
        logger.info(f"{data_zyx_c2.chunks}")

    # Counters for relabeling
    if relabeling:
        num_labels_tot = 0

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
        if channel2:
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
            model=model,
            channels=channels,
            do_3D=do_3D,
            anisotropy=anisotropy,
            label_dtype=label_dtype,
            diameter=diameter_level0 / coarsening_xy**level,
            cellprob_threshold=cellprob_threshold,
            flow_threshold=flow_threshold,
            normalize=normalize,
            min_size=min_size,
            augment=augment,
            net_avg=net_avg,
            batch_size=batch_size,
            invert=invert,
            tile=tile,
            tile_overlap=tile_overlap,
            resample=resample,
            interp=interp,
            stitch_threshold=stitch_threshold,
        )

        # Prepare keyword arguments for preprocessing function
        preprocessing_kwargs = {}
        if use_masks:
            preprocessing_kwargs = dict(
                region=region,
                current_label_path=f"{zarrurl}/labels/{output_label_name}/0",
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

        # Shift labels and update relabeling counters
        if relabeling:
            num_labels_roi = np.max(new_label_img)
            new_label_img[new_label_img > 0] += num_labels_tot
            num_labels_tot += num_labels_roi

            # Write some logs
            logger.info(f"ROI {indices}, {num_labels_roi=}, {num_labels_tot=}")

            # Check that total number of labels is under control
            if num_labels_tot > np.iinfo(label_dtype).max:
                raise ValueError(
                    "ERROR in re-labeling:"
                    f"Reached {num_labels_tot} labels, "
                    f"but dtype={label_dtype}"
                )

        if output_ROI_table:
            bbox_df = array_to_bounding_box_table(
                new_label_img,
                actual_res_pxl_sizes_zyx,
                origin_zyx=(s_z, s_y, s_x),
            )

            bbox_dataframe_list.append(bbox_df)

            overlap_list = []
            for df in bbox_dataframe_list:
                overlap_list.extend(
                    get_overlapping_pairs_3D(df, full_res_pxl_sizes_zyx)
                )
            if len(overlap_list) > 0:
                logger.warning(
                    f"{len(overlap_list)} bounding-box pairs overlap"
                )

        # Compute and store 0-th level to disk
        da.array(new_label_img).to_zarr(
            url=mask_zarr,
            region=region,
            compute=True,
        )

    logger.info(
        f"End cellpose_segmentation task for {zarrurl}, "
        "now building pyramids."
    )

    # Starting from on-disk highest-resolution data, build and write to disk a
    # pyramid of coarser levels
    build_pyramid(
        zarrurl=f"{zarrurl}/labels/{output_label_name}",
        overwrite=overwrite,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        chunksize=chunks,
        aggregation_function=np.max,
    )

    logger.info("End building pyramids")

    if output_ROI_table:
        # Handle the case where `bbox_dataframe_list` is empty (typically
        # because list_indices is also empty)
        if len(bbox_dataframe_list) == 0:
            bbox_dataframe_list = [empty_bounding_box_table()]
        # Concatenate all ROI dataframes
        df_well = pd.concat(bbox_dataframe_list, axis=0, ignore_index=True)
        df_well.index = df_well.index.astype(str)
        # Extract labels and drop them from df_well
        labels = pd.DataFrame(df_well["label"].astype(str))
        df_well.drop(labels=["label"], axis=1, inplace=True)
        # Convert all to float (warning: some would be int, in principle)
        bbox_dtype = np.float32
        df_well = df_well.astype(bbox_dtype)
        # Convert to anndata
        bbox_table = ad.AnnData(df_well, dtype=bbox_dtype)
        bbox_table.obs = labels

        # Write to zarr group
        image_group = zarr.group(f"{in_path}/{component}")
        logger.info(
            "Now writing bounding-box ROI table to "
            f"{in_path}/{component}/tables/{output_ROI_table}"
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

    return {}


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=cellpose_segmentation,
        logger_name=logger.name,
    )

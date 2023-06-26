"""
Copyright 2022 (C)
    Friedrich Miescher Institute for Biomedical Research and
    University of Zurich

    Original authors:
    Tommaso Comparin <tommaso.comparin@exact-lab.it>
    Marco Franzon <marco.franzon@exact-lab.it>
    Joel Lüthi  <joel.luethi@fmi.ch>

    This file is part of Fractal and was originally developed by eXact lab
    S.r.l.  <exact-lab.it> under contract with Liberali Lab from the Friedrich
    Miescher Institute for Biomedical Research and Pelkmans Lab from the
    University of Zurich.

Image segmentation via Cellpose library
"""
import json
import logging
import os
import time
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from typing import Sequence

import anndata as ad
import cellpose
import dask.array as da
import numpy as np
import pandas as pd
import zarr
from anndata._io.specs import write_elem
from cellpose import models
from pydantic.decorator import validate_arguments

import fractal_tasks_core
from fractal_tasks_core.lib_channels import ChannelNotFoundError
from fractal_tasks_core.lib_channels import get_channel_from_image_zarr
from fractal_tasks_core.lib_channels import OmeroChannel
from fractal_tasks_core.lib_input_models import Channel
from fractal_tasks_core.lib_masked_loading import masked_loading_wrapper
from fractal_tasks_core.lib_pyramid_creation import build_pyramid
from fractal_tasks_core.lib_regions_of_interest import (
    array_to_bounding_box_table,
)
from fractal_tasks_core.lib_regions_of_interest import (
    convert_ROI_table_to_indices,
)
from fractal_tasks_core.lib_regions_of_interest import is_ROI_table_valid
from fractal_tasks_core.lib_regions_of_interest import load_region
from fractal_tasks_core.lib_ROI_overlaps import find_overlaps_in_ROI_indices
from fractal_tasks_core.lib_ROI_overlaps import get_overlapping_pairs_3D
from fractal_tasks_core.lib_zattrs_utils import extract_zyx_pixel_sizes
from fractal_tasks_core.lib_zattrs_utils import rescale_datasets

logger = logging.getLogger(__name__)

__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__


def segment_ROI(
    x: np.ndarray,
    model: models.CellposeModel = None,
    do_3D: bool = True,
    channels=[0, 0],
    anisotropy: Optional[float] = None,
    diameter: float = 30.0,
    cellprob_threshold: float = 0.0,
    flow_threshold: float = 0.4,
    label_dtype: Optional[np.dtype] = None,
    augment: bool = False,
    net_avg: bool = False,
    min_size: int = 15,
) -> np.ndarray:
    """
    Internal function that runs Cellpose segmentation for a single ROI.

    :param x: 4D numpy array
    :param model: An instance of models.CellposeModel
    :param do_3D: If true, cellpose runs in 3D mode: runs on xy, xz & yz
                  planes, then averages the flows.
    :param channels: Which channels to use. If only one channel is provided,
                     [0, 0] should be used. If two channels are provided
                     (the first dimension of x has lenth of 2), [[1, 2]]
                     should be used (x[0, :, :, :] contains the membrane
                     channel first & x[1, :, :, :] the nuclear channel).
    :param anisotropy: Set anisotropy rescaling factor for Z dimension
    :param diameter: Expected object diameter in pixels for cellpose
    :param cellprob_threshold: Cellpose model parameter
    :param flow_threshold: Cellpose model parameter
    :param label_dtype: Label images are cast into this np.dtype
    :param augment: Whether to use cellpose augmentation to tile images
                    with overlap
    :param net_avg: Whether to use cellpose net averaging to run the 4 built-in
                    networks (useful for nuclei, cyto & cyto2, not sure it
                    works for the others)
    :param min_size: Minimum size of the segmented objects
    """

    # Write some debugging info
    logger.info(
        "[segment_ROI] START |"
        f" x: {type(x)}, {x.shape} |"
        f" {do_3D=} |"
        f" {model.diam_mean=} |"
        f" {diameter=} |"
        f" {flow_threshold=}"
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
        min_size=min_size,
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
    metadata: Dict[str, Any],
    # Task-specific arguments
    level: int,
    channel: Channel,
    channel2: Optional[Channel] = None,
    input_ROI_table: str = "FOV_ROI_table",
    output_ROI_table: Optional[str] = None,
    output_label_name: Optional[str] = None,
    use_masks: bool = True,
    relabeling: bool = True,
    # Cellpose-related arguments
    diameter_level0: float = 30.0,
    model_type: str = "cyto2",
    pretrained_model: Optional[str] = None,
    cellprob_threshold: float = 0.0,
    flow_threshold: float = 0.4,
    anisotropy: Optional[float] = None,
    min_size: int = 15,
    augment: bool = False,
    net_avg: bool = False,
    use_gpu: bool = True,
) -> Dict[str, Any]:
    """
    Run cellpose segmentation on the ROIs of a single OME-Zarr image

    :param input_paths: List of input paths where the image data is stored
                        as OME-Zarrs. Should point to the parent folder
                        containing one or many OME-Zarr files, not the
                        actual OME-Zarr file.
                        Example: ["/some/path/"]
                        This task only supports a single input path.
                        (standard argument for Fractal tasks,
                        managed by Fractal server)
    :param output_path: This parameter is not used by this task
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
                     "num_levels": int, number of pyramid levels in the image.
                     This determines how many pyramid levels are built for
                     the segmentation.
                     "coarsening_xy": int, coarsening factor in XY of the
                     downsampling when building the pyramid.
                     (standard argument for Fractal tasks,
                     managed by Fractal server)
    :param level: Pyramid level of the image to be segmented. Choose 0 to
                  process at full resolution.
    :param channel: Primary channel for segmentation; requires either
                    ``wavelength_id`` (e.g. ``A01_C01``) or ``label`` (e.g.
                    ``DAPI``).
    :param channel2: Second channel for segmentation (in the same format as
                     ``channel``). If specified, cellpose runs in dual channel
                     mode.  For dual channel segmentation of cells, the first
                     channel should contain the membrane marker, the second
                     channel should contain the nuclear marker.
    :param input_ROI_table: Name of the ROI table over which the task loops
                            to apply Cellpose segmentation.
                            Example: "FOV_ROI_table" => loop over the field of
                            views
                            "organoid_ROI_table" => loop over the organoid
                            ROI table generated by another task
                            "well_ROI_table" => process the whole well as
                            one image
    :param output_ROI_table: If provided, a ROI table with that name is
                             created, which will contain the bounding boxes of
                             the newly segmented labels. ROI tables should
                             have ``ROI`` in their name.
    :param use_masks: If ``True``, try to use masked loading and fall back
                      to ``use_masks=False`` if the ROI table is not suitable.
                      Masked loading is relevant when only a subset of the
                      bounding box should actually be processed (e.g.
                      running within organoid_ROI_table).
    :param output_label_name: Name of the output label image
                              (e.g. ``"organoids"``).
    :param relabeling: If ``True``, apply relabeling so that label values are
                       unique for all objects in the well.
    :param diameter_level0: Expected diameter of the objects that should be
                            segmented in pixels at level 0. Initial diameter
                            is rescaled using the ``level`` that was selected.
                            The rescaled value is passed as the diameter to
                            the ``CellposeModel.eval`` method.
    :param model_type: Parameter of ``CellposeModel`` class.
                       Defines which model should be used. Typical choices are
                       nuclei, cyto, cyto2 etc.
    :param pretrained_model: Parameter of ``CellposeModel`` class (takes
                             precedence over ``model_type``).
                             Allows you to specify the path of a custom
                             trained cellpose model.
    :param cellprob_threshold: Parameter of ``CellposeModel.eval`` method.
                               Valid values between -6 to 6.
                               From Cellpose documentation:
                               "Decrease this threshold if cellpose is not
                               returning as many ROIs as you’d expect.
                               Similarly, increase this threshold if cellpose
                               is returning too ROIs particularly from
                               dim areas."
    :param flow_threshold: Parameter of ``CellposeModel.eval`` method.
                           Valid values between 0.0 and 1.0.
                           From Cellpose documentation:
                           "Increase this threshold if cellpose is not
                           returning as many ROIs as you’d expect. Similarly,
                           decrease this threshold if cellpose is
                           returning too many ill-shaped ROIs."
    :param anisotropy: Ratio of the pixel sizes along Z and XY axis (ignored if
                       the image is not three-dimensional). If `None`, it is
                       inferred from the OME-NGFF metadata.
    :param min_size: Parameter of ``CellposeModel`` class.
                     Minimum size of the segmented objects (in pixels). Use -1
                     to turn off the size filter.
    :param augment: Parameter of ``CellposeModel`` class.
                    Whether to use cellpose augmentation to tile images with
                    overlap.
    :param net_avg: Parameter of ``CellposeModel`` class.
                    Whether to use cellpose net averaging to run the 4 built-in
                    networks (useful for nuclei, cyto & cyto2, not sure it
                    works for the others).
    :param use_gpu: If ``False``, always use the CPU; if ``True``, use the GPU
                    if possible (as defined in ``cellpose.core.use_gpu()``) and
                    fall-back to the CPU otherwise.
    """

    # Set input path
    if len(input_paths) > 1:
        raise NotImplementedError
    in_path = Path(input_paths[0])
    zarrurl = (in_path.resolve() / component).as_posix()
    logger.info(f"{zarrurl=}")

    # Preliminary checks on Cellpose model
    if pretrained_model is None:
        if model_type not in models.MODEL_NAMES:
            raise ValueError(f"ERROR model_type={model_type} is not allowed.")
    else:
        if not os.path.exists(pretrained_model):
            raise ValueError(f"{pretrained_model=} does not exist.")

    # Read useful parameters from metadata
    num_levels = metadata["num_levels"]
    coarsening_xy = metadata["coarsening_xy"]

    plate, well = component.split(".zarr/")

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

    # Read pixel sizes from zattrs file
    full_res_pxl_sizes_zyx = extract_zyx_pixel_sizes(
        f"{zarrurl}/.zattrs", level=0
    )
    logger.info(f"{full_res_pxl_sizes_zyx=}")
    actual_res_pxl_sizes_zyx = extract_zyx_pixel_sizes(
        f"{zarrurl}/.zattrs", level=level
    )
    logger.info(f"{actual_res_pxl_sizes_zyx=}")

    # Heuristic to determine reset_origin   # FIXME, see issue #339
    if input_ROI_table in ["FOV_ROI_table", "well_ROI_table"]:
        reset_origin = True
    else:
        reset_origin = False
    logger.info(f"{reset_origin=}")

    # Create list of indices for 3D ROIs spanning the entire Z direction
    list_indices = convert_ROI_table_to_indices(
        ROI_table,
        level=level,
        coarsening_xy=coarsening_xy,
        full_res_pxl_sizes_zyx=full_res_pxl_sizes_zyx,
        reset_origin=reset_origin,
    )

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
            # Read pixel sizes from zattrs file
            pxl_zyx = extract_zyx_pixel_sizes(
                f"{zarrurl}/.zattrs", level=level
            )
            pixel_size_z, pixel_size_y, pixel_size_x = pxl_zyx[:]
            logger.info(f"{pxl_zyx=}")
            if not np.allclose(pixel_size_x, pixel_size_y):
                raise Exception(
                    "ERROR: XY anisotropy detected"
                    f"pixel_size_x={pixel_size_x}"
                    f"pixel_size_y={pixel_size_y}"
                )
            anisotropy = pixel_size_z / pixel_size_x

    # Load zattrs file
    zattrs_file = f"{zarrurl}/.zattrs"
    with open(zattrs_file, "r") as jsonfile:
        zattrs = json.load(jsonfile)

    # Preliminary checks on multiscales
    multiscales = zattrs["multiscales"]
    if len(multiscales) > 1:
        raise NotImplementedError(
            f"Found {len(multiscales)} multiscales, "
            "but only one is currently supported."
        )
    if "coordinateTransformations" in multiscales[0].keys():
        raise NotImplementedError(
            "global coordinateTransformations at the multiscales "
            "level are not currently supported"
        )

    # Rescale datasets (only relevant for level>0)
    if not multiscales[0]["axes"][0]["name"] == "c":
        raise ValueError(
            "Cannot set `remove_channel_axis=True` for multiscale "
            f'metadata with axes={multiscales[0]["axes"]}. '
            'First axis should have name "c".'
        )
    new_datasets = rescale_datasets(
        datasets=multiscales[0]["datasets"],
        coarsening_xy=coarsening_xy,
        reference_level=level,
        remove_channel_axis=True,
    )

    # Write zattrs for labels and for specific label
    new_labels = [output_label_name]
    try:
        with open(f"{zarrurl}/labels/.zattrs", "r") as f_zattrs:
            existing_labels = json.load(f_zattrs)["labels"]
    except FileNotFoundError:
        existing_labels = []
    intersection = set(new_labels) & set(existing_labels)
    logger.info(f"{new_labels=}")
    logger.info(f"{existing_labels=}")
    if intersection:
        raise RuntimeError(
            f"Labels {intersection} already exist but are also part of outputs"
        )
    labels_group = zarr.group(f"{zarrurl}/labels")
    labels_group.attrs["labels"] = existing_labels + new_labels

    label_group = labels_group.create_group(output_label_name)
    label_group.attrs["image-label"] = {"version": __OME_NGFF_VERSION__}
    label_group.attrs["multiscales"] = [
        {
            "name": output_label_name,
            "version": __OME_NGFF_VERSION__,
            "axes": [
                ax for ax in multiscales[0]["axes"] if ax["type"] != "channel"
            ],
            "datasets": new_datasets,
        }
    ]

    # Open new zarr group for mask 0-th level
    zarr.group(f"{zarrurl}/labels")
    zarr.group(f"{zarrurl}/labels/{output_label_name}")
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
            min_size=min_size,
            augment=augment,
            net_avg=net_avg,
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
            logger.info(
                f"ROI {indices}, " f"{num_labels_roi=}, " f"{num_labels_tot=}"
            )

            # Check that total number of labels is under control
            if num_labels_tot > np.iinfo(label_dtype).max:
                raise Exception(
                    "ERROR in re-labeling:"
                    f"Reached {num_labels_tot} labels, "
                    f"but dtype={label_dtype}"
                )

        if output_ROI_table:
            bbox_df = array_to_bounding_box_table(
                new_label_img, actual_res_pxl_sizes_zyx
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
        overwrite=False,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        chunksize=chunks,
        aggregation_function=np.max,
    )

    logger.info("End building pyramids")

    if output_ROI_table:
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
        group_tables = zarr.group(f"{in_path}/{component}/tables/")
        write_elem(group_tables, output_ROI_table, bbox_table)
        logger.info(
            "Bounding box ROI table written to "
            f"{in_path}/{component}/tables/{output_ROI_table}"
        )

        # WARNING: the following OME-NGFF metadata are based on a proposed
        # change to the specs (https://github.com/ome/ngff/pull/64)

        # Update OME-NGFF metadata for tables group
        current_tables = group_tables.attrs.asdict().get("tables") or []
        if output_ROI_table in current_tables:
            # FIXME: move this check to an earlier stage of the task
            raise ValueError(
                f"{in_path}/{component}/tables/ already includes "
                f"{output_ROI_table=} in {current_tables=}"
            )
        new_tables = current_tables + [output_ROI_table]
        group_tables.attrs["tables"] = new_tables

        # Update OME-NGFF metadata for current-table group
        bbox_table_group = zarr.group(
            f"{in_path}/{component}/tables/{output_ROI_table}"
        )
        bbox_table_group.attrs["type"] = "ngff:region_table"
        bbox_table_group.attrs["region"] = {
            "path": f"../labels/{output_label_name}"
        }
        bbox_table_group.attrs["instance_key"] = "label"

    return {}


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=cellpose_segmentation,
        logger_name=logger.name,
    )

# Copyright 2022-2026 (C) BioVisionCenter, University of Zurich
"""
Task for threshold-based segmentation of OME-Zarr images.
"""

import logging
import time

import numpy as np
from ngio import OmeZarrContainer, open_ome_zarr_container
from ngio.experimental.iterators import MaskedSegmentationIterator, SegmentationIterator
from ngio.images._image import _parse_channel_selection
from ngio.images._masked_image import MaskedImage
from ngio.utils import NgioValueError
from pydantic import validate_call

from fractal_tasks_core._threshold_segmentation_utils import (
    AnyCreateRoiTableModel,
    CreateMaskingRoiTable,
    InputChannel,
    IteratorConfiguration,
    MaskingConfiguration,
    OtsuConfiguration,
    PrePostProcessConfiguration,
    SegmentationConfiguration,
    SkipCreateMaskingRoiTable,
    segmentation_function,
)

logger = logging.getLogger("threshold_segmentation")


def load_masked_image(
    ome_zarr: OmeZarrContainer,
    masking_configuration: MaskingConfiguration,
    level_path: str | None = None,
) -> MaskedImage:
    """Load a masked image from an OME-Zarr based on the masking configuration.

    Args:
        ome_zarr: The OME-Zarr container.
        masking_configuration (MaskingConfiguration): Configuration for masking.
        level_path (str | None): Optional path to a specific resolution level.

    """
    if masking_configuration.mode == "Table Name":
        masking_table_name = masking_configuration.identifier
        masking_label_name = None
    else:
        masking_label_name = masking_configuration.identifier
        masking_table_name = None
    logger.info(f"Using masking with {masking_table_name=}, {masking_label_name=}")

    # Base Iterator with masking
    masked_image = ome_zarr.get_masked_image(
        masking_label_name=masking_label_name,
        masking_table_name=masking_table_name,
        path=level_path,
    )
    return masked_image


def _format_label_name(label_name_template: str, channel_identifier: str) -> str:
    """Format the label name based on the provided template and channel identifier.

    Args:
        label_name_template (str): The template for the label name. This
        might contain a placeholder "{channel_identifier}" which will be replaced
        by the channel identifier or no placeholder at all,
        in which case the channel identifier will be ignored.
        channel_identifier (str): The channel identifier to insert into the
            label name template.

    Returns:
        str: The formatted label name.
    """
    try:
        label_name = label_name_template.format(channel_identifier=channel_identifier)
    except KeyError as e:
        raise ValueError(
            "Label Name format error only allowed placeholder is "
            f"'channel_identifier'. {{{e}}} was provided."
        ) from e
    return label_name


def _skip_segmentation(channels: InputChannel, ome_zarr: OmeZarrContainer) -> bool:
    """Check whether to skip the current task based on the channel configuration.

    If the channel selection specified in the channels parameter is not
    valid for the provided OME-Zarr image, this function checks the
    skip_if_missing attribute of the channels configuration.
    If skip_if_missing is True, the function returns True, indicating that the task
    should be skipped. If skip_if_missing is False, a ValueError is raised.

    Args:
        channels (InputChannel): The channel selection configuration.
        ome_zarr (OmeZarrContainer): The OME-Zarr container to check against.

    Returns:
        bool: True if the task should be skipped due to missing channels,
        False otherwise.

    """
    channel_selection = channels.to_channel_selection_models()
    image = ome_zarr.get_image()
    try:
        _parse_channel_selection(image=image, channel_selection=channel_selection)
    except NgioValueError as e:
        if channels.skip_if_missing:
            logger.warning(
                f"Channel selection {channel_selection} is not valid for the provided "
                "image, but skip_if_missing is set to True. Skipping segmentation."
            )
            logger.debug(f"Original error message: {e}")
            return True
        else:
            raise ValueError(
                f"Channel selection {channel_selection} is not valid for the provided "
                "image. If you want to skip processing when channels are missing, "
                "set skip_if_missing to True."
            ) from e
    return False


@validate_call
def threshold_segmentation(
    *,
    # Fractal managed parameters
    zarr_url: str,
    # Segmentation parameters
    channels: InputChannel,
    label_name: str = "{channel_identifier}_segmented",
    level_path: str | None = None,
    method: SegmentationConfiguration = OtsuConfiguration(),
    # Iteration parameters
    iterator_configuration: IteratorConfiguration | None = None,
    pre_post_process: PrePostProcessConfiguration = PrePostProcessConfiguration(),  # noqa: B008
    create_masking_roi_table: AnyCreateRoiTableModel = SkipCreateMaskingRoiTable(),  # noqa: B008
    overwrite: bool = True,
) -> None:
    """Segment an image using intensity thresholding.

    Pixels above the threshold are treated as foreground and connected
    components are labelled. The threshold can be computed automatically
    (Otsu) or set manually.

    Args:
        zarr_url (str): URL to the OME-Zarr container.
        channels (InputChannel): Channel to use for segmentation,
            selected by label, wavelength ID, or index.
        label_name (str): Name of the resulting label image. Optionally, it can contain
            a placeholder "{channel_identifier}" which will be replaced by the
            channel identifier specified in the channels parameter.
        level_path (str | None): If the OME-Zarr has multiple resolution levels,
            the level to use can be specified here. If not provided, the highest
            resolution level will be used.
        iterator_configuration (IteratorConfiguration | None): Configuration
            for the segmentation iterator. This can be used to specify masking
            and/or a ROI table.
        method (SegmentationConfiguration): Configuration for the segmentation method.
        pre_post_process (PrePostProcessConfiguration): Configuration for pre- and
            post-processing steps.
        create_masking_roi_table (AnyCreateRoiTableModel): Configuration to
            create a masking ROI table after segmentation.
        overwrite (bool): Whether to overwrite an existing label image.
            Defaults to True.
    """
    # Use the first of input_paths
    logger.info(f"{zarr_url=}")

    # Open the OME-Zarr container
    ome_zarr = open_ome_zarr_container(zarr_url)
    logger.info(f"{ome_zarr=}")
    # Validate that the specified channels are present in the image
    if _skip_segmentation(channels=channels, ome_zarr=ome_zarr):
        return None

    # Format the label name based on the provided template and channel identifier
    label_name = _format_label_name(
        label_name_template=label_name, channel_identifier=channels.identifier
    )
    logger.info(f"Formatted label name: {label_name=}")

    # Derive the label and an get it at the specified level path
    ome_zarr.derive_label(name=label_name, overwrite=overwrite)
    label = ome_zarr.get_label(name=label_name, path=level_path)
    logger.info(f"Derived label image: {label=}")

    # Set up the appropriate iterator based on the configuration
    if iterator_configuration is None:
        iterator_configuration = IteratorConfiguration()

    axes_order = "czyx" if ome_zarr.is_3d else "cyx"
    logger.info(f"Segmenting using {axes_order=}")

    if iterator_configuration.masking is None:
        # Create a basic SegmentationIterator without masking
        image = ome_zarr.get_image(path=level_path)
        logger.info(f"{image=}")
        iterator = SegmentationIterator(
            input_image=image,
            output_label=label,
            channel_selection=channels.to_channel_selection_models(),
            axes_order=axes_order,
        )
    else:
        # Since masking is requested, we need to determine load a masking image
        masked_image = load_masked_image(
            ome_zarr=ome_zarr,
            masking_configuration=iterator_configuration.masking,
            level_path=level_path,
        )
        logger.info(f"{masked_image=}")
        # A masked iterator is created instead of a basic segmentation iterator
        # This will do two major things:
        # 1) It will iterate only over the regions of interest defined by the
        #   masking table or label image
        # 2) It will only write the segmentation results within the masked regions
        iterator = MaskedSegmentationIterator(
            input_image=masked_image,
            output_label=label,
            channel_selection=channels.to_channel_selection_models(),
            axes_order=axes_order,
        )
    # Make sure that if we have a time axis, we iterate over it
    # Strict=False means that if there no z axis or z is size 1, it will still work
    # If your segmentation needs requires a volume, use strict=True
    iterator = iterator.by_zyx(strict=False)
    logger.info(f"Iterator created: {iterator=}")

    if iterator_configuration.roi_table is not None:
        # If a ROI table is provided, we load it and use it to further restrict
        # the iteration to the ROIs defined in the table
        # Be aware that this is not an alternative to masking
        # but only an additional restriction
        table = ome_zarr.get_generic_roi_table(name=iterator_configuration.roi_table)
        logger.info(f"ROI table retrieved: {table=}")
        iterator = iterator.product(table)
        logger.info(f"Iterator updated with ROI table: {iterator=}")

    # Keep track of the maximum label to ensure unique across iterations
    max_label = 0
    #
    # Core processing loop
    #
    logger.info("Starting processing...")
    run_times = []
    num_rois = len(iterator.rois)
    logging_step = max(1, num_rois // 10)
    for it, (image_data, writer) in enumerate(iterator.iter_as_numpy()):
        start_time = time.time()
        label_img = segmentation_function(
            image_data=image_data,
            method=method,
            pre_post_process=pre_post_process,
        )
        # Ensure unique labels across different chunks
        label_img = np.where(label_img == 0, 0, label_img + max_label)
        max_label = max(max_label, label_img.max())
        writer(label_img)
        iteration_time = time.time() - start_time
        run_times.append(iteration_time)

        # Only log the progress every logging_step iterations
        if it % logging_step == 0 or it == num_rois - 1:
            avg_time = sum(run_times) / len(run_times)
            logger.info(
                f"Processed ROI {it + 1}/{num_rois} "
                f"(avg time per ROI: {avg_time:.2f} s)"
            )
    logger.info(f"label {label_name} successfully created at {zarr_url}")

    # Building a masking roi table
    if isinstance(create_masking_roi_table, CreateMaskingRoiTable):
        table_name = create_masking_roi_table.get_table_name(label_name=label_name)
        masking_roi_table = label.build_masking_roi_table()
        ome_zarr.add_table(
            name=table_name, table=masking_roi_table, overwrite=overwrite
        )
    return None


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=threshold_segmentation,
        logger_name=logger.name,
    )

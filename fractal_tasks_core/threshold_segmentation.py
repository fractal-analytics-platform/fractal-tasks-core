# Copyright 2022-2026 (C) BioVisionCenter, University of Zurich
"""
Task for threshold-based segmentation of OME-Zarr images.
"""

import logging

from fractal_tasks_utils.segmentation import (
    IteratorConfig,
    compute_segmentation,
    setup_segmentation_iterator,
)
from fractal_tasks_utils.segmentation._transforms import SegmentationTransformConfig
from ngio import OmeZarrContainer, open_ome_zarr_container
from ngio.images._image import _parse_channel_selection
from ngio.utils import NgioValueError
from pydantic import validate_call

from fractal_tasks_core._threshold_segmentation_utils import (
    AnyCreateRoiTableModel,
    InputChannel,
    OtsuConfiguration,
    SegmentationConfiguration,
    SkipCreateMaskingRoiTable,
    segmentation_function,
)
from fractal_tasks_core._utils import format_template_name

logger = logging.getLogger("threshold_segmentation")


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
    output_label_name: str = "{channel_identifier}_segmented",
    level_path: str | None = None,
    method: SegmentationConfiguration = OtsuConfiguration(),
    # Iteration parameters
    iterator_configuration: IteratorConfig | None = None,
    pre_post_process: SegmentationTransformConfig = SegmentationTransformConfig(),  # noqa: B008
    create_masking_roi_table: AnyCreateRoiTableModel = SkipCreateMaskingRoiTable(),  # noqa: B008
    overwrite: bool = True,
) -> None:
    """Segment an image using intensity thresholding.

    Pixels above the threshold are treated as foreground and connected
    components are labelled. The threshold can be computed automatically
    (Otsu) or set manually.

    Args:
        zarr_url: URL to the OME-Zarr container.
        channels: Channel to use for segmentation,
            selected by label, wavelength ID, or index.
        output_label_name: Name of the resulting label image. Optionally, it can
            contain a placeholder "{channel_identifier}" which will be replaced by the
            channel identifier specified in the channels parameter.
        level_path: If the OME-Zarr has multiple resolution levels,
            the level to use can be specified here. If not provided, the highest
            resolution level will be used.
        iterator_configuration: Optionally restrict segmentation to a specific
            set of ROIs or a sub-region. If not provided, the full image is
            segmented.
        method: Configuration for the segmentation method.
        pre_post_process: Configuration for pre- and post-processing transforms
            applied by the iterator.
        create_masking_roi_table: Configuration to create a masking ROI table
            after segmentation.
        overwrite: Whether to overwrite an existing label image.
            Defaults to True.
    """
    logger.info(f"{zarr_url=}")

    # Open the OME-Zarr container for early channel validation
    ome_zarr = open_ome_zarr_container(zarr_url)
    logger.info(f"{ome_zarr=}")
    if _skip_segmentation(channels=channels, ome_zarr=ome_zarr):
        return None

    # Format the label name based on the provided template and channel identifier
    output_label_name = format_template_name(
        output_label_name, channel_identifier=channels.identifier
    )
    logger.info(f"Formatted label name: {output_label_name=}")

    # Set up iterator (opens ome_zarr internally, derives label, handles masking)
    iterator = setup_segmentation_iterator(
        zarr_url=zarr_url,
        channels=[channels.to_channel_selection_models()],
        output_label_name=output_label_name,
        level_path=level_path,
        iterator_configuration=iterator_configuration,
        segmentation_transform_config=pre_post_process,
        overwrite=overwrite,
    )

    # Run the core segmentation loop
    compute_segmentation(
        segmentation_func=lambda x: segmentation_function(input_image=x, method=method),
        iterator=iterator,
    )
    logger.info(f"label {output_label_name} successfully created at {zarr_url}")

    # Build a masking ROI table if requested
    create_masking_roi_table.create(
        ome_zarr=ome_zarr, label_name=output_label_name, overwrite=overwrite
    )
    return None


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=threshold_segmentation,
        logger_name=logger.name,
    )

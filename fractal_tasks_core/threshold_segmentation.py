# Copyright 2022-2026 (C) BioVisionCenter, University of Zurich
"""
Task for threshold-based segmentation of OME-Zarr images.
"""

import logging

from fractal_tasks_utils.segmentation import (
    IteratorConfiguration,
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
    CreateMaskingRoiTable,
    InputChannel,
    OtsuConfiguration,
    SegmentationConfiguration,
    SkipCreateMaskingRoiTable,
    segmentation_function,
)

logger = logging.getLogger("threshold_segmentation")


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
    pre_post_process: SegmentationTransformConfig = SegmentationTransformConfig(),  # noqa: B008
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
        pre_post_process (SegmentationTransformConfig): Configuration for pre- and
            post-processing transforms applied by the iterator.
        create_masking_roi_table (AnyCreateRoiTableModel): Configuration to
            create a masking ROI table after segmentation.
        overwrite (bool): Whether to overwrite an existing label image.
            Defaults to True.
    """
    logger.info(f"{zarr_url=}")

    # Open the OME-Zarr container for early channel validation
    ome_zarr = open_ome_zarr_container(zarr_url)
    logger.info(f"{ome_zarr=}")
    if _skip_segmentation(channels=channels, ome_zarr=ome_zarr):
        return None

    # Format the label name based on the provided template and channel identifier
    label_name = _format_label_name(
        label_name_template=label_name, channel_identifier=channels.identifier
    )
    logger.info(f"Formatted label name: {label_name=}")

    # Set up iterator (opens ome_zarr internally, derives label, handles masking)
    iterator = setup_segmentation_iterator(
        zarr_url=zarr_url,
        channels=[channels.to_channel_selection_models()],
        label_name=label_name,
        level_path=level_path,
        iterator_configuration=iterator_configuration,
        segmentation_transform_config=pre_post_process,
        overwrite=overwrite,
    )

    # Run the core segmentation loop
    compute_segmentation(
        func=lambda x: segmentation_function(input_image=x, method=method),
        iterator=iterator,
    )
    logger.info(f"label {label_name} successfully created at {zarr_url}")

    # Build a masking ROI table if requested
    if isinstance(create_masking_roi_table, CreateMaskingRoiTable):
        table_name = create_masking_roi_table.get_table_name(label_name=label_name)
        label_img = ome_zarr.get_label(name=label_name)
        masking_roi_table = label_img.build_masking_roi_table()
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

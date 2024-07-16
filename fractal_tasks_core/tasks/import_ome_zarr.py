# Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Tommaso Comparin <tommaso.comparin@exact-lab.it>
#
# This file is part of Fractal and was originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
"""
Task to import an existing OME-Zarr.
"""
import logging
from typing import Any
from typing import Optional

import dask.array as da
import zarr
from pydantic import validate_call

from fractal_tasks_core.channels import update_omero_channels
from fractal_tasks_core.ngff import detect_ome_ngff_type
from fractal_tasks_core.ngff import NgffImageMeta
from fractal_tasks_core.roi import get_image_grid_ROIs
from fractal_tasks_core.roi import get_single_image_ROI
from fractal_tasks_core.tables import write_table

logger = logging.getLogger(__name__)


def _process_single_image(
    image_path: str,
    add_image_ROI_table: bool,
    add_grid_ROI_table: bool,
    update_omero_metadata: bool,
    *,
    grid_YX_shape: Optional[tuple[int, int]] = None,
    overwrite: bool = False,
) -> dict[str, str]:
    """
    Validate OME-NGFF metadata and optionally generate ROI tables.

    This task:

    1. Validates OME-NGFF image metadata, via `NgffImageMeta`;
    2. Optionally generates and writes two ROI tables;
    3. Optionally update OME-NGFF omero metadata.
    4. Returns dataset types

    Args:
        image_path: Absolute path to the image Zarr group.
        add_image_ROI_table: Whether to add a `image_ROI_table` table
            (argument propagated from `import_ome_zarr`).
        add_grid_ROI_table: Whether to add a `grid_ROI_table` table (argument
            propagated from `import_ome_zarr`).
        update_omero_metadata: Whether to update Omero-channels metadata
            (argument propagated from `import_ome_zarr`).
        grid_YX_shape: YX shape of the ROI grid (it must be not `None`, if
            `add_grid_ROI_table=True`.
    """

    # Note from zarr docs: `r+` means read/write (must exist)
    image_group = zarr.open_group(image_path, mode="r+")
    image_meta = NgffImageMeta(**image_group.attrs.asdict())

    # Preliminary checks
    if add_grid_ROI_table and (grid_YX_shape is None):
        raise ValueError(
            f"_process_single_image called with {add_grid_ROI_table=}, "
            f"but {grid_YX_shape=}."
        )

    pixels_ZYX = image_meta.get_pixel_sizes_zyx(level=0)

    # Read zarr array
    dataset_subpath = image_meta.datasets[0].path
    array = da.from_zarr(f"{image_path}/{dataset_subpath}")

    # Prepare image_ROI_table and write it into the zarr group
    if add_image_ROI_table:
        image_ROI_table = get_single_image_ROI(array.shape, pixels_ZYX)
        write_table(
            image_group,
            "image_ROI_table",
            image_ROI_table,
            overwrite=overwrite,
            table_attrs={"type": "roi_table"},
        )

    # Prepare grid_ROI_table and write it into the zarr group
    if add_grid_ROI_table:
        grid_ROI_table = get_image_grid_ROIs(
            array.shape,
            pixels_ZYX,
            grid_YX_shape,
        )
        write_table(
            image_group,
            "grid_ROI_table",
            grid_ROI_table,
            overwrite=overwrite,
            table_attrs={"type": "roi_table"},
        )

    # Update Omero-channels metadata
    if update_omero_metadata:
        # Extract number of channels from zarr array
        try:
            channel_axis_index = image_meta.axes_names.index("c")
        except ValueError:
            logger.error(f"Existing axes: {image_meta.axes_names}")
            msg = (
                "OME-Zarrs with no channel axis are not currently "
                "supported in fractal-tasks-core. Upcoming flexibility "
                "improvements are tracked in https://github.com/"
                "fractal-analytics-platform/fractal-tasks-core/issues/150."
            )
            logger.error(msg)
            raise NotImplementedError(msg)
        logger.info(f"Existing axes: {image_meta.axes_names}")
        logger.info(f"Channel-axis index: {channel_axis_index}")
        num_channels_zarr = array.shape[channel_axis_index]
        logger.info(
            f"{num_channels_zarr} channel(s) found in Zarr array "
            f"at {image_path}/{dataset_subpath}"
        )
        # Update or create omero channels metadata
        old_omero = image_group.attrs.get("omero", {})
        old_channels = old_omero.get("channels", [])
        if len(old_channels) > 0:
            logger.info(
                f"{len(old_channels)} channel(s) found in NGFF omero metadata"
            )
            if len(old_channels) != num_channels_zarr:
                error_msg = (
                    "Channels-number mismatch: Number of channels in the "
                    f"zarr array ({num_channels_zarr}) differs from number "
                    "of channels listed in NGFF omero metadata "
                    f"({len(old_channels)})."
                )
                logging.error(error_msg)
                raise ValueError(error_msg)
        else:
            old_channels = [{} for ind in range(num_channels_zarr)]
        new_channels = update_omero_channels(old_channels)
        new_omero = old_omero.copy()
        new_omero["channels"] = new_channels
        image_group.attrs.update(omero=new_omero)

    # Determine image types:
    # Later: also provide a has_T flag.
    # TODO: Potentially also load acquisition metadata if available in a Zarr
    is_3D = False
    if "z" in image_meta.axes_names:
        if array.shape[-3] > 1:
            is_3D = True
    types = dict(is_3D=is_3D)
    return types


@validate_call
def import_ome_zarr(
    *,
    # Fractal parameters
    zarr_urls: list[str],
    zarr_dir: str,
    # Core parameters
    zarr_name: str,
    update_omero_metadata: bool = True,
    add_image_ROI_table: bool = True,
    add_grid_ROI_table: bool = True,
    # Advanced parameters
    grid_y_shape: int = 2,
    grid_x_shape: int = 2,
    overwrite: bool = False,
) -> dict[str, Any]:
    """
    Import a single OME-Zarr into Fractal.

    The single OME-Zarr can be a full OME-Zarr HCS plate or an individual
    OME-Zarr image. The image needs to be in the zarr_dir as specified by the
    dataset. The current version of this task:

    1. Creates the appropriate components-related metadata, needed for
       processing an existing OME-Zarr through Fractal.
    2. Optionally adds new ROI tables to the existing OME-Zarr.

    Args:
        zarr_urls: List of paths or urls to the individual OME-Zarr image to
            be processed. Not used.
            (standard argument for Fractal tasks, managed by Fractal server).
        zarr_dir: path of the directory where the new OME-Zarrs will be
            created.
            (standard argument for Fractal tasks, managed by Fractal server).
        zarr_name: The OME-Zarr name, without its parent folder. The parent
            folder is provided by zarr_dir; e.g. `zarr_name="array.zarr"`,
            if the OME-Zarr path is in `/zarr_dir/array.zarr`.
        add_image_ROI_table: Whether to add a `image_ROI_table` table to each
            image, with a single ROI covering the whole image.
        add_grid_ROI_table: Whether to add a `grid_ROI_table` table to each
            image, with the image split into a rectangular grid of ROIs.
        grid_y_shape: Y shape of the ROI grid in `grid_ROI_table`.
        grid_x_shape: X shape of the ROI grid in `grid_ROI_table`.
        update_omero_metadata: Whether to update Omero-channels metadata, to
            make them Fractal-compatible.
        overwrite: Whether new ROI tables (added when `add_image_ROI_table`
            and/or `add_grid_ROI_table` are `True`) can overwite existing ones.
    """

    # Is this based on the Zarr_dir or the zarr_urls?
    if len(zarr_urls) > 0:
        logger.warning(
            "Running import while there are already items from the image list "
            "provided to the task. The following inputs were provided: "
            f"{zarr_urls=}"
            "This task will not process the existing images, but look for "
            f"zarr files named {zarr_name=} in the {zarr_dir=} instead."
        )

    zarr_path = f"{zarr_dir.rstrip('/')}/{zarr_name}"
    logger.info(f"Zarr path: {zarr_path}")

    root_group = zarr.open_group(zarr_path, mode="r")
    ngff_type = detect_ome_ngff_type(root_group)
    grid_YX_shape = (grid_y_shape, grid_x_shape)

    image_list_updates = []
    if ngff_type == "plate":
        for well in root_group.attrs["plate"]["wells"]:
            well_path = well["path"]

            well_group = zarr.open_group(zarr_path, path=well_path, mode="r")
            for image in well_group.attrs["well"]["images"]:
                image_path = image["path"]
                zarr_url = f"{zarr_path}/{well_path}/{image_path}"
                types = _process_single_image(
                    zarr_url,
                    add_image_ROI_table,
                    add_grid_ROI_table,
                    update_omero_metadata,
                    grid_YX_shape=grid_YX_shape,
                    overwrite=overwrite,
                )
                image_list_updates.append(
                    dict(
                        zarr_url=zarr_url,
                        attributes=dict(
                            plate=zarr_name,
                            well=well_path.replace("/", ""),
                        ),
                        types=types,
                    )
                )
    elif ngff_type == "well":
        logger.warning(
            "Only OME-Zarr for plates are fully supported in Fractal; "
            f"e.g. the current one ({ngff_type=}) cannot be "
            "processed via the `maximum_intensity_projection` task."
        )
        for image in root_group.attrs["well"]["images"]:
            image_path = image["path"]
            zarr_url = f"{zarr_path}/{image_path}"
            well_name = "".join(zarr_path.split("/")[-2:])
            types = _process_single_image(
                zarr_url,
                add_image_ROI_table,
                add_grid_ROI_table,
                update_omero_metadata,
                grid_YX_shape=grid_YX_shape,
                overwrite=overwrite,
            )
            image_list_updates.append(
                dict(
                    zarr_url=zarr_url,
                    attributes=dict(
                        well=well_name,
                    ),
                    types=types,
                )
            )
    elif ngff_type == "image":
        logger.warning(
            "Only OME-Zarr for plates are fully supported in Fractal; "
            f"e.g. the current one ({ngff_type=}) cannot be "
            "processed via the `maximum_intensity_projection` task."
        )
        zarr_url = zarr_path
        types = _process_single_image(
            zarr_url,
            add_image_ROI_table,
            add_grid_ROI_table,
            update_omero_metadata,
            grid_YX_shape=grid_YX_shape,
            overwrite=overwrite,
        )
        image_list_updates.append(
            dict(
                zarr_url=zarr_url,
                types=types,
            )
        )

    image_list_changes = dict(image_list_updates=image_list_updates)
    return image_list_changes


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=import_ome_zarr,
        logger_name=logger.name,
    )

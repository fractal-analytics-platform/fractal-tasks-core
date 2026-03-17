# Copyright 2022-2026 (C) BioVisionCenter, University of Zurich
"""
Task to import an existing OME-Zarr.
"""

import logging
from typing import Any

from ngio import (
    OmeZarrContainer,
    OmeZarrPlate,
    OmeZarrWell,
    Roi,
    open_ome_zarr_container,
    open_ome_zarr_plate,
    open_ome_zarr_well,
)
from ngio.tables import RoiTable
from pydantic import validate_call

from fractal_tasks_core._utils import AVAILABLE_TABLE_BACKENDS, DEFAULT_TABLE_BACKEND

logger = logging.getLogger("import_ome_zarr")


def _build_xy_roi_table(
    ome_zarr_image: OmeZarrContainer,
    grid_YX_shape: tuple[int, int] | None = None,
    backend: AVAILABLE_TABLE_BACKENDS = DEFAULT_TABLE_BACKEND,
    overwrite: bool = False,
) -> None:
    """
    Build a grid_ROI_table with a rectangular grid of ROIs covering the whole image.

    Args:
        ome_zarr_image: OME-Zarr image container.
        grid_YX_shape: YX shape of the ROI grid.
        overwrite: Whether to overwrite an existing `grid_ROI_table`.
    """
    if grid_YX_shape is None:
        raise ValueError("grid_YX_shape cannot be None when building grid ROI table.")
    image = ome_zarr_image.get_image()
    pixel_size = image.pixel_size
    image_shape = image.shape
    image_YX_shape = image_shape[-2:]
    roi_id = 0
    rois = []
    z_start = 0
    z_ind = image.axes_handler.get_index("z")
    z_length = image_shape[z_ind] if z_ind is not None else 1

    t_start = 0
    t_ind = image.axes_handler.get_index("t")
    t_length = image_shape[t_ind] if t_ind is not None else 1

    for y in range(0, image_YX_shape[0], grid_YX_shape[0]):
        for x in range(0, image_YX_shape[1], grid_YX_shape[1]):
            x_length = min(grid_YX_shape[1], image_YX_shape[1] - x)
            y_length = min(grid_YX_shape[0], image_YX_shape[0] - y)
            roi_pixels = Roi.from_values(
                name=f"ROI_{roi_id}",
                slices={
                    "x": (x, x_length),
                    "y": (y, y_length),
                    "z": (z_start, z_length),
                    "t": (t_start, t_length),
                },
                space="pixel",
            )
            roi_world = roi_pixels.to_world(pixel_size=pixel_size)
            rois.append(roi_world)
            roi_id += 1
    table = RoiTable(rois=rois)
    ome_zarr_image.add_table(
        name="grid_ROI_table", table=table, overwrite=overwrite, backend=backend
    )


def _process_single_image(
    *,
    zarr_path: str,
    ome_zarr_image: OmeZarrContainer,
    add_image_roi_table: bool,
    add_grid_roi_table: bool,
    update_omero_metadata: bool,
    grid_YX_shape: tuple[int, int] | None = None,
    attributes: dict[str, Any] | None = None,
    table_backend: AVAILABLE_TABLE_BACKENDS = DEFAULT_TABLE_BACKEND,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    """
    Optionally generate ROI tables and update omero metadata for a single image.

    Args:
        zarr_path: Absolute path to the image Zarr group.
        ome_zarr_image: OME-Zarr image container.
        add_image_roi_table: Whether to add an `image_ROI_table` table.
        add_grid_roi_table: Whether to add a `grid_ROI_table` table.
        update_omero_metadata: Whether to update Omero-channels metadata.
        grid_YX_shape: YX shape of the ROI grid (must not be `None` when
            `add_grid_roi_table=True`).
        attributes: Optional image attributes to include in the update dict.
        table_backend: Backend to use for the new ROI tables.
        overwrite: Whether to overwrite existing ROI tables.
    """
    if add_image_roi_table:
        table = ome_zarr_image.build_image_roi_table()
        ome_zarr_image.add_table(
            name="image_ROI_table",
            table=table,
            overwrite=overwrite,
            backend=table_backend,
        )

    if add_grid_roi_table:
        _build_xy_roi_table(
            ome_zarr_image=ome_zarr_image,
            grid_YX_shape=grid_YX_shape,
            overwrite=overwrite,
            backend=table_backend,
        )

    if update_omero_metadata:
        image = ome_zarr_image.get_image()
        channel_names = image.channel_labels
        wavelengths = image.channels_meta.channel_wavelength_ids
        ome_zarr_image.set_channel_meta(
            labels=channel_names,
            wavelength_id=wavelengths,
        )

    type_updates = {
        "is_3D": ome_zarr_image.is_3d,
        # TODO add this after v1 testing is removed
        # "is_time_series": ome_zarr_image.is_time_series,
    }
    image_list_update = {
        "zarr_url": zarr_path,
        "types": type_updates,
    }
    if attributes is not None:
        image_list_update["attributes"] = attributes
    return [image_list_update]


def _process_well(
    *,
    zarr_path: str,
    ome_zarr_well: OmeZarrWell,
    add_image_roi_table: bool,
    add_grid_roi_table: bool,
    update_omero_metadata: bool,
    grid_YX_shape: tuple[int, int] | None = None,
    table_backend: AVAILABLE_TABLE_BACKENDS = DEFAULT_TABLE_BACKEND,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    """
    For each image in the well, create an image list update dict.
    """
    image_list_updates = []
    *_, row, column = zarr_path.rstrip("/").split("/")
    attributes = {
        "well": f"{row}{int(column):02d}",
    }
    for path in ome_zarr_well.paths():
        ome_zarr_image = ome_zarr_well.get_image(path)
        image_zarr_path = f"{zarr_path}/{path}"
        _updates = _process_single_image(
            zarr_path=image_zarr_path,
            ome_zarr_image=ome_zarr_image,
            add_image_roi_table=add_image_roi_table,
            add_grid_roi_table=add_grid_roi_table,
            update_omero_metadata=update_omero_metadata,
            grid_YX_shape=grid_YX_shape,
            attributes=attributes,
            table_backend=table_backend,
            overwrite=overwrite,
        )
        image_list_updates.extend(_updates)
    return image_list_updates


def _process_plate(
    *,
    zarr_path: str,
    ome_zarr_plate: OmeZarrPlate,
    add_image_roi_table: bool,
    add_grid_roi_table: bool,
    update_omero_metadata: bool,
    grid_YX_shape: tuple[int, int] | None = None,
    table_backend: AVAILABLE_TABLE_BACKENDS = DEFAULT_TABLE_BACKEND,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    """For each image in the plate, create an image list update dict."""
    image_list_updates = []
    plate_name = zarr_path.rstrip("/").split("/")[-1]
    for path, image in ome_zarr_plate.get_images().items():
        row, column, _ = path.split("/")
        well_name = f"{row}{int(column):02d}"
        attributes = {
            "plate": plate_name,
            "well": well_name,
        }
        image_zarr_path = f"{zarr_path}/{path}"
        _updates = _process_single_image(
            zarr_path=image_zarr_path,
            ome_zarr_image=image,
            add_image_roi_table=add_image_roi_table,
            add_grid_roi_table=add_grid_roi_table,
            update_omero_metadata=update_omero_metadata,
            grid_YX_shape=grid_YX_shape,
            attributes=attributes,
            overwrite=overwrite,
            table_backend=table_backend,
        )
        image_list_updates.extend(_updates)
    return image_list_updates


def open_unknown_container(
    zarr_path: str,
) -> OmeZarrContainer | OmeZarrWell | OmeZarrPlate:
    """
    Detect the OME-NGFF type of the OME-Zarr, based on its root metadata.

    The OME-NGFF type can be "plate", "well" or "image". If the OME-Zarr does
    not contain valid OME-NGFF metadata, an error is raised.

    Args:
        zarr_path: Path to the OME-Zarr.
    Returns:
        OmeZarrContainer, OmeZarrWell, or OmeZarrPlate
    """
    errors = []
    try:
        ome_zarr = open_ome_zarr_container(zarr_path)
        return ome_zarr
    except Exception as e:
        errors.append(e)

    try:
        ome_zarr = open_ome_zarr_plate(zarr_path)
        return ome_zarr
    except Exception as e:
        errors.append(e)

    try:
        ome_zarr = open_ome_zarr_well(zarr_path)
        return ome_zarr
    except Exception as e:
        errors.append(e)

    base_error = f"Could not detect OME-NGFF type of OME-Zarr at {zarr_path}."
    error_messages = "\n".join([f"{type(e).__name__}: {str(e)}" for e in errors])
    raise ValueError(f"{base_error}\nErrors:\n{error_messages}")


@validate_call
def import_ome_zarr(
    *,
    # Fractal parameters
    zarr_dir: str,
    # Core parameters
    zarr_name: str,
    update_omero_metadata: bool = True,
    add_image_roi_table: bool = True,
    add_grid_roi_table: bool = True,
    # Advanced parameters
    grid_y_shape: int = 2,
    grid_x_shape: int = 2,
    table_backend: AVAILABLE_TABLE_BACKENDS = DEFAULT_TABLE_BACKEND,
    # Other parameters
    overwrite: bool = False,
) -> dict[str, Any]:
    """
    Import a single OME-Zarr into Fractal.

    The single OME-Zarr can be a full OME-Zarr HCS plate or an individual
    OME-Zarr image. The image needs to be in the zarr_dir as specified by the
    dataset. This task registers the OME-Zarr with Fractal so it can be used
    in processing workflows, and optionally adds new ROI tables to the existing
    OME-Zarr.

    Args:
        zarr_dir: Path of the directory where the OME-Zarr is located
            (standard argument for Fractal tasks, managed by Fractal server).
        zarr_name: The OME-Zarr name, without its parent folder. The parent
            folder is provided by zarr_dir; e.g. `zarr_name="array.zarr"`,
            if the OME-Zarr path is in `/zarr_dir/array.zarr`.
        add_image_roi_table: Whether to add a `image_ROI_table` table to each
            image, with a single ROI covering the whole image.
        add_grid_roi_table: Whether to add a `grid_ROI_table` table to each
            image, with the image split into a rectangular grid of ROIs.
        grid_y_shape: Y shape of the ROI grid in `grid_ROI_table`.
        grid_x_shape: X shape of the ROI grid in `grid_ROI_table`.
        table_backend: Backend to use for the new ROI tables. Defaults to "anndata".
        update_omero_metadata: Whether to update Omero-channels metadata, to
            make them Fractal-compatible.
        overwrite: Whether new ROI tables (added when `add_image_roi_table`
            and/or `add_grid_roi_table` are `True`) can overwrite existing ones.
    """

    zarr_path = f"{zarr_dir.rstrip('/')}/{zarr_name}"
    logger.info(f"Zarr path: {zarr_path}")

    ome_zarr = open_unknown_container(zarr_path)
    image_list_updates = []
    if isinstance(ome_zarr, OmeZarrPlate):
        image_list_updates = _process_plate(
            zarr_path=zarr_path,
            ome_zarr_plate=ome_zarr,
            add_image_roi_table=add_image_roi_table,
            add_grid_roi_table=add_grid_roi_table,
            update_omero_metadata=update_omero_metadata,
            grid_YX_shape=(grid_y_shape, grid_x_shape),
            table_backend=table_backend,
            overwrite=overwrite,
        )
    elif isinstance(ome_zarr, OmeZarrWell):
        image_list_updates = _process_well(
            zarr_path=zarr_path,
            ome_zarr_well=ome_zarr,
            add_image_roi_table=add_image_roi_table,
            add_grid_roi_table=add_grid_roi_table,
            update_omero_metadata=update_omero_metadata,
            grid_YX_shape=(grid_y_shape, grid_x_shape),
            table_backend=table_backend,
            overwrite=overwrite,
        )
    elif isinstance(ome_zarr, OmeZarrContainer):
        image_list_updates = _process_single_image(
            zarr_path=zarr_path,
            ome_zarr_image=ome_zarr,
            add_image_roi_table=add_image_roi_table,
            add_grid_roi_table=add_grid_roi_table,
            update_omero_metadata=update_omero_metadata,
            grid_YX_shape=(grid_y_shape, grid_x_shape),
            table_backend=table_backend,
            overwrite=overwrite,
        )
    else:
        raise ValueError(
            f"Unexpected OME-NGFF type for OME-Zarr at {zarr_path}: "
            f"{type(ome_zarr).__name__}"
        )

    image_list_changes = {
        "image_list_updates": image_list_updates,
    }
    return image_list_changes


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=import_ome_zarr)

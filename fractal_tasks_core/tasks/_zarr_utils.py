import copy
import logging

import anndata as ad
import zarr
from filelock import FileLock

from fractal_tasks_core.ngff.zarr_utils import load_NgffWellMeta
from fractal_tasks_core.tables import write_table
from fractal_tasks_core.tables.v1 import get_tables_list_v1
from fractal_tasks_core.utils import _split_well_path_image_path

logger = logging.getLogger(__name__)


def _copy_hcs_ome_zarr_metadata(
    zarr_url_origin: str,
    zarr_url_new: str,
) -> None:
    """
    Updates the necessary metadata for a new copy of an OME-Zarr image

    Based on an existing OME-Zarr image in the same well, the metadata is
    copied and added to the new zarr well. Additionally, the well-level
    metadata is updated to include this new image.

    Args:
        zarr_url_origin: zarr_url of the origin image
        zarr_url_new: zarr_url of the newly created image. The zarr-group
            already needs to exist, but metadata is written by this function.
    """
    # Copy over OME-Zarr metadata for illumination_corrected image
    # See #681 for discussion for validation of this zattrs
    old_image_group = zarr.open_group(zarr_url_origin, mode="r")
    old_attrs = old_image_group.attrs.asdict()
    zarr_url_new = zarr_url_new.rstrip("/")
    new_image_group = zarr.group(zarr_url_new)
    new_image_group.attrs.put(old_attrs)

    # Update well metadata about adding the new image:
    new_image_path = zarr_url_new.split("/")[-1]
    well_url, old_image_path = _split_well_path_image_path(zarr_url_origin)
    _update_well_metadata(well_url, old_image_path, new_image_path)


def _update_well_metadata(
    well_url: str,
    old_image_path: str,
    new_image_path: str,
    timeout: int = 120,
) -> None:
    """
    Update the well metadata by adding the new_image_path to the image list.

    The content of new_image_path will be based on old_image_path, the origin
    for the new image that was created.
    This function aims to avoid race conditions with other processes that try
    to update the well metadata file by using FileLock & Timeouts

    Args:
        well_url: Path to the HCS OME-Zarr well that needs to be updated
        old_image_path: path relative to well_url where the original image is
            found
        new_image_path: path relative to well_url where the new image is placed
        timeout: Timeout in seconds for trying to get the file lock
    """
    lock = FileLock(f"{well_url}/.zattrs.lock")
    with lock.acquire(timeout=timeout):
        well_meta = load_NgffWellMeta(well_url)
        existing_well_images = [image.path for image in well_meta.well.images]
        if new_image_path in existing_well_images:
            raise ValueError(
                f"Could not add the {new_image_path=} image to the well "
                "metadata because and image with that name "
                f"already existed in the well metadata: {well_meta}"
            )
        try:
            well_meta_image_old = next(
                image
                for image in well_meta.well.images
                if image.path == old_image_path
            )
        except StopIteration:
            raise ValueError(
                f"Could not find an image with {old_image_path=} in the "
                "current well metadata."
            )
        well_meta_image = copy.deepcopy(well_meta_image_old)
        well_meta_image.path = new_image_path
        well_meta.well.images.append(well_meta_image)
        well_meta.well.images = sorted(
            well_meta.well.images,
            key=lambda _image: _image.path,
        )

        well_group = zarr.group(well_url)
        well_group.attrs.put(well_meta.model_dump(exclude_none=True))

    # One could catch the timeout with a try except Timeout. But what to do
    # with it?


def _split_base_suffix(input: str) -> tuple[str, str]:
    parts = input.split("_")
    base = parts[0]
    if len(parts) > 1:
        suffix = "_".join(parts[1:])
    else:
        suffix = ""
    return base, suffix


def _get_matching_ref_acquisition_path_heuristic(
    path_list: list[str], path: str
) -> str:
    """
    Pick the best match from path_list to a given path

    This is a workaround to find the reference registration acquisition when
    there are multiple OME-Zarrs with the same acquisition identifier in the
    well metadata and we need to find which one is the reference for a given
    path.

    Args:
        path_list: List of paths to OME-Zarr images in the well metadata. For
            example: ['0', '0_illum_corr']
        path: A given path for which we want to find the reference image. For
            example, '1_illum_corr'

    Returns:
        The best matching reference path. If no direct match is found, it
        returns the most similar one based on suffix hierarchy or the base
        path if applicable. For example, '0_illum_corr' with the example
        inputs above.
    """

    # Extract the base number and suffix from the input path
    base, suffix = _split_base_suffix(path)

    # Sort path_list
    sorted_path_list = sorted(path_list)

    # First matching rule: a path with the same suffix
    for p in sorted_path_list:
        # Split the list path into base and suffix
        p_base, p_suffix = _split_base_suffix(p)
        # If suffices match, it's the match.
        if p_suffix == suffix:
            return p

    # If no match is found, return the first entry in the list
    logger.warning(
        "No heuristic reference acquisition match found, defaulting to first "
        f"option {sorted_path_list[0]}."
    )
    return sorted_path_list[0]


def _copy_tables_from_zarr_url(
    origin_zarr_url: str,
    target_zarr_url: str,
    table_type: str = None,
    overwrite: bool = True,
) -> None:
    """
    Copies all ROI tables from one Zarr into a new Zarr

    Args:
        origin_zarr_url: url of the OME-Zarr image that contains tables.
            e.g. /path/to/my_plate.zarr/B/03/0
        target_zarr_url: url of the new OME-Zarr image where tables are copied
            to. e.g. /path/to/my_plate.zarr/B/03/0_illum_corr
        table_type: Filter for specific table types that should be copied.
        overwrite: Whether existing tables of the same name in the
            target_zarr_url should be overwritten.
    """
    table_list = get_tables_list_v1(
        zarr_url=origin_zarr_url, table_type=table_type
    )

    if table_list:
        logger.info(
            f"Copying the tables {table_list} from {origin_zarr_url} to "
            f"{target_zarr_url}."
        )
        new_image_group = zarr.group(target_zarr_url)

        for table in table_list:
            logger.info(f"Copying table: {table}")
            # Get the relevant metadata of the Zarr table & add it
            table_url = f"{origin_zarr_url}/tables/{table}"
            old_table_group = zarr.open_group(table_url, mode="r")
            # Write the Zarr table
            curr_table = ad.read_zarr(table_url)
            write_table(
                new_image_group,
                table,
                curr_table,
                table_attrs=old_table_group.attrs.asdict(),
                overwrite=overwrite,
            )

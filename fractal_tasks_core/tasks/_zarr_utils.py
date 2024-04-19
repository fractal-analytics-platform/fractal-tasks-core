import copy

import zarr
from filelock import FileLock

from fractal_tasks_core.ngff.zarr_utils import load_NgffWellMeta


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
    zarr_url_new = zarr_url_new.rstrip("/")
    group = zarr.open_group(zarr_url_origin, mode="r")
    new_attrs = group.attrs.asdict()
    new_image_group = zarr.group(zarr_url_new)
    new_image_group.attrs.put(new_attrs)

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
            key=lambda x: x.path,
        )

        well_group = zarr.group(well_url)
        well_group.attrs.put(well_meta.dict(exclude_none=True))

    # One could catch the timeout with a try except Timeout. But what to do
    # with it?


def _split_well_path_image_path(zarr_url: str) -> tuple[str, str]:
    """
    Returns path to well folder for HCS OME-Zarr `zarr_url`.
    """
    zarr_url = zarr_url.rstrip("/")
    well_path = "/".join(zarr_url.split("/")[:-1])
    img_path = zarr_url.split("/")[-1]
    return well_path, img_path

"""
Utilities to work with the Pydantic models from `specs.py` for Zarr groups.
"""
import logging

import zarr.hierarchy
from zarr.errors import GroupNotFoundError

from fractal_tasks_core.ngff.specs import NgffImageMeta
from fractal_tasks_core.ngff.specs import NgffPlateMeta
from fractal_tasks_core.ngff.specs import NgffWellMeta

logger = logging.getLogger(__name__)


class ZarrGroupNotFoundError(ValueError):
    """
    Wrap zarr.errors.GroupNotFoundError

    This is used to provide a user-friendly error message.
    """

    pass


def load_NgffImageMeta(zarr_path: str) -> NgffImageMeta:
    """
    Load the attributes of a zarr group and cast them to `NgffImageMeta`.

    Args:
        zarr_path: Path to the zarr group.

    Returns:
        A new `NgffImageMeta` object.
    """
    try:
        zarr_group = zarr.open_group(zarr_path, mode="r")
    except GroupNotFoundError:
        error_msg = (
            "Could not load attributes for the requested image, "
            f"because no Zarr group was found at {zarr_path}"
        )
        logging.error(error_msg)
        raise ZarrGroupNotFoundError(error_msg)
    zarr_attrs = zarr_group.attrs.asdict()
    try:
        return NgffImageMeta(**zarr_attrs)
    except Exception as e:
        logging.error(
            f"Contents of {zarr_path} cannot be cast to NgffImageMeta.\n"
            f"Original error:\n{str(e)}"
        )
        raise e


def load_NgffWellMeta(zarr_path: str) -> NgffWellMeta:
    """
    Load the attributes of a zarr group and cast them to `NgffWellMeta`.

    Args:
        zarr_path: Path to the zarr group.

    Returns:
        A new `NgffWellMeta` object.
    """
    try:
        zarr_group = zarr.open_group(zarr_path, mode="r")
    except GroupNotFoundError:
        error_msg = (
            "Could not load attributes for the requested well, "
            f"because no Zarr group was found at {zarr_path}"
        )
        logging.error(error_msg)
        raise ZarrGroupNotFoundError(error_msg)
    zarr_attrs = zarr_group.attrs.asdict()
    try:
        return NgffWellMeta(**zarr_attrs)
    except Exception as e:
        logging.error(
            f"Contents of {zarr_path} cannot be cast to NgffWellMeta.\n"
            f"Original error:\n{str(e)}"
        )
        raise e


def load_NgffPlateMeta(zarr_path: str) -> NgffPlateMeta:
    """
    Load the attributes of a zarr group and cast them to `NgffPlateMeta`.

    Args:
        zarr_path: Path to the zarr group.

    Returns:
        A new `NgffPlateMeta` object.
    """
    try:
        zarr_group = zarr.open_group(zarr_path, mode="r")
    except GroupNotFoundError:
        error_msg = (
            "Could not load attributes for the requested plate, "
            f"because no Zarr group was found at {zarr_path}"
        )
        logging.error(error_msg)
        raise ZarrGroupNotFoundError(error_msg)
    zarr_attrs = zarr_group.attrs.asdict()
    try:
        return NgffPlateMeta(**zarr_attrs)
    except Exception as e:
        logging.error(
            f"Contents of {zarr_path} cannot be cast to NgffPlateMeta.\n"
            f"Original error:\n{str(e)}"
        )
        raise e


def detect_ome_ngff_type(group: zarr.hierarchy.Group) -> str:
    """
    Given a Zarr group, find whether it is an OME-NGFF plate, well or image.

    Args:
        group: Zarr group

    Returns:
        The detected OME-NGFF type (`plate`, `well` or `image`).
    """
    attrs = group.attrs.asdict()
    if "plate" in attrs.keys():
        ngff_type = "plate"
    elif "well" in attrs.keys():
        ngff_type = "well"
    elif "multiscales" in attrs.keys():
        ngff_type = "image"
    else:
        error_msg = (
            "Zarr group at cannot be identified as one "
            "of OME-NGFF plate/well/image groups."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    logger.info(f"Zarr group identified as OME-NGFF {ngff_type}.")
    return ngff_type

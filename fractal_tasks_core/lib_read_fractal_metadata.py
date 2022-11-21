from pathlib import Path
from typing import Any
from typing import Dict
from typing import Sequence
from typing import Union

import zarr


def find_omengff_acquisition(image_zarr_path: Path) -> Union[int, None]:
    """
    Discover the acquisition index based on OME-NGFF metadata

    Given the path to a zarr image folder (e.g. ``/path/plate.zarr/B/03/0``),
    extract the acquisition index from the ``.zattrs`` file of the parent
    folder (i.e. at the well level), or return ``None`` if acquisition is not
    specified.

    Notes:

    1. For non-multiplexing datasets, acquisition is not a required
       information in the metadata. If it is not there, this function
       returns ``None``.
    2. This function fails if we use an image that does not belong to
       an OME-NGFF well.

    :param image_zarr_path: full path to an OME-NGFF image folder
    """

    # Identify well plate and attrs
    well_zarr_path = image_zarr_path.parent
    if not (well_zarr_path / ".zattrs").exists():
        raise ValueError(
            f"{str(well_zarr_path)} must be an OME-NGFF plate "
            "folder, but it does not include a .zattrs file."
        )
    well_group = zarr.open_group(str(well_zarr_path))
    attrs_images = well_group.attrs["well"]["images"]

    # Loook for the acqusition of the current image (if any)
    acquisition = None
    for img_dict in attrs_images:
        if (
            img_dict["path"] == image_zarr_path.name
            and "acquisition" in img_dict.keys()
        ):
            acquisition = img_dict["acquisition"]
            break

    return acquisition


def get_parameters_from_metadata(
    *,
    keys: Sequence[str],
    metadata: Dict[str, Any],
    image_zarr_path: Path,
) -> Dict[str, Any]:
    """
    Flexibly extract parameters from metadata dictionary

    This covers both parameters which are acquisition-specific (if the image
    belongs to an OME-NGFF array and its acquisition is specified) or simply
    available in the dictionary.
    The two cases are handled as::

        metadata[acquisition]["some_parameter"]  # acquisition available
        metadata["some_parameter"]               # acquisition not available



    :param keys: list of required parameters
    :param metadata: metadata dictionary
    :param image_zarr_path: full path to image, e.g.
                             ``/path/plate.zarr/B/03/0``
    """

    parameters = {}
    acquisition = find_omengff_acquisition(image_zarr_path)
    if acquisition is not None:
        parameters["acquisition"] = acquisition

    for key in keys:
        if acquisition is None:
            parameter = metadata[key]
        else:
            try:
                parameter = metadata[key][acquisition]
            except TypeError:
                parameter = metadata[key]
            except KeyError:
                parameter = metadata[key]
        parameters[key] = parameter
    return parameters

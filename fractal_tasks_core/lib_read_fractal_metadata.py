from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Sequence

import zarr


def discover_acquisition_from_ome_zarr(
    well_zarr_path: Path, component: str
) -> int:
    """
    FIXME: write docstring

    : param well_zarr_path: TBD
    : param component: TBD
    """
    well_path = Path(component).parent
    image_path = Path(component).name
    well_group = zarr.open_group(str(well_zarr_path.parent / well_path))
    try:
        acquisition = next(
            item["acquisition"]
            for item in well_group.attrs["well"]["images"]
            if item["path"] == image_path
        )
        return acquisition
    except KeyError:
        return None


def get_parameter_from_metadata(
    keys: Sequence[str],
    metadata: Dict[str, Any],
    well_zarr_path: Path,
    component: str,
) -> List:

    """
    FIXME: write docstring

    : param keys: TBD
    : param metadata: TBD
    : param well_zarr_path: TBD
    : param component: TBD
    """

    parameters = []
    acquisition = discover_acquisition_from_ome_zarr(well_zarr_path, component)

    for key in keys:
        try:
            parameter = metadata[key][acquisition]
        except KeyError:
            parameter = metadata[key]
        except TypeError:
            parameter = metadata[key]
        parameters.append(parameter)
    return parameters

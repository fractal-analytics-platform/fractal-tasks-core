from typing import Dict
from typing import Literal
from typing import Optional
from typing import Tuple

from pydantic import BaseModel
from pydantic import Field

from fractal_tasks_core.channels import OmeroChannel


class InitArgsRegistration(BaseModel):
    """
    Registration init args.

    Passed from `image_based_registration_hcs_init` to
    `calculate_registration_image_based`.

    Attributes:
        reference_zarr_url: zarr_url for the reference image
    """

    reference_zarr_url: str


class InitArgsRegistrationConsensus(BaseModel):
    """
    Registration consensus init args.

    Provides the list of zarr_urls for all acquisitions for a given well

    Attributes:
        zarr_url_list: List of zarr_urls for all the OME-Zarr images in the
            well.
    """

    zarr_url_list: list[str]


class InitArgsCellVoyager(BaseModel):
    """
    Arguments to be passed from cellvoyager converter init to compute

    Attributes:
        image_dir: Directory where the raw images are found
        plate_prefix: part of the image filename needed for finding the
            right subset of image files
        well_ID: part of the image filename needed for finding the
            right subset of image files
        image_extension: part of the image filename needed for finding the
            right subset of image files
        include_glob_patterns: Additional glob patterns to filter the available
            images with.
        exclude_glob_patterns: Glob patterns to exclude.
        acquisition: Acquisition metadata needed for multiplexing
    """

    image_dir: str
    plate_prefix: str
    well_ID: str
    image_extension: str
    include_glob_patterns: Optional[list[str]] = None
    exclude_glob_patterns: Optional[list[str]] = None
    acquisition: Optional[int] = None


class InitArgsIllumination(BaseModel):
    """
    Dummy model description.

    Attributes:
        raw_path: dummy attribute description.
        subsets: dummy attribute description.
    """

    raw_path: str
    subsets: dict[Literal["C_index"], int] = Field(default_factory=dict)


class InitArgsMIP(BaseModel):
    """
    Init Args for MIP task.

    Attributes:
        origin_url: Path to the zarr_url with the 3D data
        method: Projection method to be used. See `DaskProjectionMethod`
        overwrite: If `True`, overwrite the task output.
        new_plate_name: Name of the new OME-Zarr HCS plate
    """

    origin_url: str
    method: str
    overwrite: bool
    new_plate_name: str


class MultiplexingAcquisition(BaseModel):
    """
    Input class for Multiplexing Cellvoyager converter

    Attributes:
        image_dir: Path to the folder that contains the Cellvoyager image
            files for that acquisition and the MeasurementData &
            MeasurementDetail metadata files.
        allowed_channels: A list of `OmeroChannel` objects, where each channel
            must include the `wavelength_id` attribute and where the
            `wavelength_id` values must be unique across the list.
    """

    image_dir: str
    allowed_channels: list[OmeroChannel]


class ChunkSizes(BaseModel):
    """
    Chunk size settings for OME-Zarrs.

    Attributes:
        t: Chunk size of time axis.
        c: Chunk size of channel axis.
        z: Chunk size of Z axis.
        y: Chunk size of y axis.
        x: Chunk size of x axis.
    """

    t: Optional[int] = None
    c: Optional[int] = 1
    z: Optional[int] = 10
    y: Optional[int] = None
    x: Optional[int] = None

    def get_chunksize(
        self, chunksize_default: Optional[Dict[str, int]] = None
    ) -> Tuple[int, ...]:
        # Define the valid keys
        valid_keys = {"t", "c", "z", "y", "x"}

        # If chunksize_default is not None, check for invalid keys
        if chunksize_default:
            invalid_keys = set(chunksize_default.keys()) - valid_keys
            if invalid_keys:
                raise ValueError(
                    f"Invalid keys in chunksize_default: {invalid_keys}. "
                    f"Only {valid_keys} are allowed."
                )

        # Filter and use only valid keys from chunksize_default
        chunksize = {
            key: chunksize_default[key]
            for key in valid_keys
            if chunksize_default and key in chunksize_default
        }

        # Overwrite with the values from the ChunkSizes instance if they are
        # not None
        for key in valid_keys:
            if getattr(self, key) is not None:
                chunksize[key] = getattr(self, key)

        # Ensure the output tuple is ordered and matches the tczyx structure
        ordered_keys = ["t", "c", "z", "y", "x"]
        return tuple(
            chunksize[key] for key in ordered_keys if key in chunksize
        )

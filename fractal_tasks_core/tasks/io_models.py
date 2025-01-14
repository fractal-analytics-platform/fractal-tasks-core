from typing import Dict
from typing import Literal
from typing import Optional
from typing import Tuple

from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator
from typing_extensions import Self

from fractal_tasks_core.channels import ChannelInputModel
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


class NapariWorkflowsOutput(BaseModel):
    """
    A value of the `output_specs` argument in `napari_workflows_wrapper`.

    Attributes:
        type: Output type (either `label` or `dataframe`).
        label_name: Label name (for label outputs, it is used as the name of
            the label; for dataframe outputs, it is used to fill the
            `region["path"]` field).
        table_name: Table name (for dataframe outputs only).
    """

    type: Literal["label", "dataframe"]
    label_name: str
    table_name: Optional[str] = None

    @model_validator(mode="after")
    def table_name_only_for_dataframe_type(self: Self) -> Self:
        """
        Check that table_name is set only for dataframe outputs.
        """
        _type = self.type
        _table_name = self.table_name
        if (_type == "dataframe" and (not _table_name)) or (
            _type != "dataframe" and _table_name
        ):
            raise ValueError(
                f"Output item has type={_type} but table_name={_table_name}."
            )
        return self


class NapariWorkflowsInput(BaseModel):
    """
    A value of the `input_specs` argument in `napari_workflows_wrapper`.

    Attributes:
        type: Input type (either `image` or `label`).
        label_name: Label name (for label inputs only).
        channel: `ChannelInputModel` object (for image inputs only).
    """

    type: Literal["image", "label"]
    label_name: Optional[str] = None
    channel: Optional[ChannelInputModel] = None

    @model_validator(mode="after")
    def label_name_is_present(self: Self) -> Self:
        """
        Check that label inputs have `label_name` set.
        """
        label_name = self.label_name
        _type = self.type
        if _type == "label" and label_name is None:
            raise ValueError(
                f"Input item has type={_type} but label_name={label_name}."
            )
        return self

    @model_validator(mode="after")
    def channel_is_present(self: Self) -> Self:
        """
        Check that image inputs have `channel` set.
        """
        _type = self.type
        channel = self.channel
        if _type == "image" and channel is None:
            raise ValueError(
                f"Input item has type={_type} but channel={channel}."
            )
        return self


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

from collections.abc import Iterator
from pathlib import Path
from typing import Literal, Optional, Tuple

from pydantic import BaseModel, Field, field_validator


class Window(BaseModel):
    """
    Custom class for Omero-channel window, based on OME-NGFF v0.4.

    Attributes:
        min: Do not change. It will be set to `0` by default.
        max:
            Do not change. It will be set according to bit-depth of the images
            by default (e.g. 65535 for 16 bit images).
        start: Lower-bound rescaling value for visualization.
        end: Upper-bound rescaling value for visualization.
    """

    min: Optional[int] = None
    max: Optional[int] = None
    start: int
    end: int


class OmeroChannel(BaseModel):
    """
    Custom class for Omero channels, based on OME-NGFF v0.4.

    Attributes:
        wavelength_id: Unique ID for the channel wavelength, e.g. `A01_C01`.
        index: Do not change. For internal use only.
        label: Name of the channel.
        window: Optional `Window` object to set default display settings. If
            unset, it will be set to the full bit range of the image
            (e.g. 0-65535 for 16 bit images).
        color: Optional hex colormap to display the channel in napari (it
            must be of length 6, e.g. `00FFFF`).
        active: Should this channel be shown in the viewer?
        coefficient: Do not change. Omero-channel attribute.
        inverted: Do not change. Omero-channel attribute.
    """

    # Custom

    wavelength_id: str
    index: Optional[int] = None

    # From OME-NGFF v0.4 transitional metadata

    label: Optional[str] = None
    window: Optional[Window] = None
    color: Optional[str] = None
    active: bool = True
    coefficient: int = 1
    inverted: bool = False

    @field_validator("color", mode="after")
    @classmethod
    def valid_hex_color(cls, v: Optional[str]) -> Optional[str]:
        """
        Check that `color` is made of exactly six elements which are letters
        (a-f or A-F) or digits (0-9).
        """
        if v is None:
            return v
        if len(v) != 6:
            raise ValueError(f'color must have length 6 (given: "{v}")')
        allowed_characters = "abcdefABCDEF0123456789"
        for character in v:
            if character not in allowed_characters:
                raise ValueError(
                    "color must only include characters from "
                    f'"{allowed_characters}" (given: "{v}")'
                )
        return v


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


class ProfileCorrectionModel(BaseModel):
    """
    Parameters for profile-based corrections.

    Attributes:
        model: The correction model to be applied.
        folder: Path of folder of correction profiles.
        profiles: Dictionary where keys match the `wavelength_id`
            attributes of existing channels (e.g.  `A01_C01` ) and values are
            the filenames of the corresponding correction profiles.
    """

    model_config = {"title": "Correction with Profiles"}
    model: Literal["Profile"] = "Profile"
    folder: str
    profiles: dict[str, str]

    def items(
        self,
    ) -> Iterator[Tuple[str, str],]:
        root_path = Path(self.folder)
        for wavelength_id, profile_name in self.profiles.items():
            yield wavelength_id, (root_path / profile_name).as_posix()


class ConstantCorrectionModel(BaseModel):
    """
    Parameters for constant-based corrections.

    Attributes:
        model: The correction model to be applied.
        constants: Dictionary where keys match the `wavelength_id`
            attributes of existing channels (e.g.  `A01_C01` ) and values are
            the constant values to be used for correction.
    """

    model_config = {"title": "Correction with constants"}
    model: Literal["Constant"] = "Constant"
    constants: dict[str, int]


class NoCorrectionModel(BaseModel):
    """
    Select for no correction to be applied.

    Attributes:
        model: The correction model to be applied.

    """

    model_config = {"title": "No Correction"}
    model: Literal["No Correction"] = "No Correction"

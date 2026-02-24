from collections.abc import Iterator
from pathlib import Path
from typing import Literal, Optional, Tuple

from pydantic import BaseModel, Field


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

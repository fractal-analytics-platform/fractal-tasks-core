from typing import Literal
from typing import Optional

from pydantic import BaseModel
from pydantic import Field


class InitArgsRegistration(BaseModel):
    """
    Dummy model description.

    Attributes:
        ref_path: dummy attribute description.
    """

    ref_path: str


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
        image_glob_patterns: Additional glob patterns to filter the available
            images with
        acquisition: Acquisition metadata needed for multiplexing
    """

    image_dir: str
    plate_prefix: str
    well_ID: str
    image_extension: str
    image_glob_patterns: Optional[list[str]]
    acquisition: Optional[int]


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
    Dummy model description.

    Attributes:
        new_path: dummy attribute description.
        new_plate: dummy attribute description.
    """

    new_path: str
    new_plate: str  # FIXME: remove this

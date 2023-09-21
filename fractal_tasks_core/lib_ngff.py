"""
Pydantic models related to OME-NGFF 0.4 specs.
"""
import logging
from enum import Enum
from typing import Literal
from typing import Optional
from typing import Union

import zarr
from pydantic import BaseModel
from pydantic import Field
from pydantic import validator


class Version(Enum):
    field_0_4 = "0.4"


class Window(BaseModel):
    """
    FIXME: specify that here we deviate from specs
    """

    end: Optional[float] = None
    max: float
    min: float
    start: Optional[float] = None


class Channel(BaseModel):
    window: Window
    label: Optional[str] = None
    family: Optional[str] = None
    color: str
    active: Optional[bool] = None


class Omero(BaseModel):
    channels: list[Channel]


class Axe(BaseModel):
    name: str
    type: Optional[str] = None  # or maybe Literal["channel", "time", "space"]


class CoordinateTransformation(BaseModel):
    type: Literal["scale"]
    scale: list[float] = Field(..., min_items=2)


class Type2(Enum):
    translation = "translation"


class CoordinateTransformation1(BaseModel):
    type: Type2
    translation: list[float] = Field(..., min_items=2)


class Dataset(BaseModel):
    path: str
    coordinateTransformations: list[
        Union[CoordinateTransformation, CoordinateTransformation1]
    ] = Field(  # noqa
        ..., min_items=1
    )

    @property
    def scale_transformation(self):
        _transformations = [
            t for t in self.coordinateTransformations if t.type == "scale"
        ]
        if len(_transformations) == 0:
            raise ValueError(
                "Missing scale transformation in dataset.\n"
                "Current coordinateTransformations:\n"
                f"{self.coordinateTransformations}"
            )
        elif len(_transformations) > 1:
            raise ValueError(
                "More than one scale transformation in dataset.\n"
                "Current coordinateTransformations:\n"
                f"{self.coordinateTransformations}"
            )
        else:
            return _transformations[0]


class Multiscale(BaseModel):
    name: Optional[str] = None
    datasets: list[Dataset] = Field(..., min_items=1)
    version: Optional[Version] = None
    axes: list[Axe] = Field(..., max_items=5, min_items=2, unique_items=True)
    coordinateTransformations: Optional[
        list[Union[CoordinateTransformation, CoordinateTransformation1]]
    ] = None

    @validator("coordinateTransformations", always=True)
    def _no_global_coordinateTransformations(cls, v):
        if v is not None:
            raise NotImplementedError(
                "Global coordinateTransformations at the multiscales "
                "level are not currently supported."
            )


class NgffImageMeta(BaseModel):
    multiscales: list[Multiscale] = Field(
        ...,
        description="The multiscale datasets for this image",
        min_items=1,
        unique_items=True,
    )
    omero: Optional[Omero] = None

    @property
    def multiscale(self) -> Multiscale:
        if len(self.multiscales) > 1:
            raise NotImplementedError(
                "Only images with one multiscale are supported "
                f"(given: {len(self.multiscales)}"
            )
        return self.multiscales[0]

    @property
    def datasets(self) -> list[Dataset]:
        return self.multiscale.datasets

    @property
    def num_levels(self) -> int:
        return len(self.datasets)

    @property
    def axes(self) -> list[str]:
        return [ax.name for ax in self.multiscale.axes]

    @property
    def pixel_sizes_zyx(self) -> list[tuple[float, float, float]]:
        x_index = self.axes.index("x")
        y_index = self.axes.index("y")
        try:
            z_index = self.axes.index("z")
        except ValueError:
            z_index = None
            logging.warning(
                f"Z axis is not present (axes: {self.axes}), and Z pixel "
                "size is set to 1. This may work, by accident, but it is "
                "not fully supported."
            )
        _pixel_sizes_zyx = []
        for level in range(self.num_levels):
            scale = self.datasets[level].scale_transformation.scale
            pixel_size_x = scale[x_index]
            pixel_size_y = scale[y_index]
            if z_index is not None:
                pixel_size_z = scale[z_index]
            else:
                pixel_size_z = 1.0
            _pixel_sizes_zyx.append((pixel_size_z, pixel_size_y, pixel_size_x))
            if min(_pixel_sizes_zyx[-1]) < 1e-9:
                raise ValueError(
                    f"Pixel sizes at level {level} are too small: "
                    f"{_pixel_sizes_zyx[-1]}"
                )

        return _pixel_sizes_zyx

    def get_pixel_sizes_zyx(
        self, *, level: int = 0
    ) -> tuple[float, float, float]:
        return self.pixel_sizes_zyx[level]

    @property
    def coarsening_xy(self) -> int:
        current_ratio = None
        for ind in range(1, self.num_levels):
            ratio_x = round(
                self.pixel_sizes_zyx[ind][2] / self.pixel_sizes_zyx[ind - 1][2]
            )
            ratio_y = round(
                self.pixel_sizes_zyx[ind][1] / self.pixel_sizes_zyx[ind - 1][1]
            )
            if ratio_x != ratio_y:
                raise NotImplementedError(
                    "Inhomogeneous coarsening in X/Y directions "
                    "is not supported.\n"
                    f"ZYX pixel sizes:\n {self.pixel_sizes_zyx}"
                )
            if current_ratio is None:
                current_ratio = ratio_x
            else:
                if current_ratio != ratio_x:
                    raise NotImplementedError(
                        "Inhomogeneous coarsening across levels "
                        "is not supported.\n"
                        f"ZYX pixel sizes:\n {self.pixel_sizes_zyx}"
                    )

        return current_ratio


class Image(BaseModel):
    """
    FIXME: restore spec for `path` (currently changed from
    `constr(regex=r'^[A-Za-z0-9]+$')` to `str`) via validator
    """

    acquisition: Optional[int] = Field(
        None, description="A unique identifier within the context of the plate"
    )
    path: str = Field(
        ..., description="The path for this field of view subgroup"
    )


class Well(BaseModel):
    images: list[Image] = Field(
        ...,
        description="The images included in this well",
        min_items=1,
        unique_items=True,
    )
    version: Optional[Version] = Field(
        None, description="The version of the specification"
    )


class NgffWellMeta(BaseModel):
    well: Optional[Well] = None

    def get_acquisition_paths(self) -> dict[int, str]:
        """
        Create mapping from acquisition indices to corresponding paths.

        Runs on the well zarr attributes and loads the relative paths in the
        well.

        Returns:
            Dictionary with `(acquisition index: image path)` key/value pairs.
        """
        acquisition_dict = {}
        for image in self.well.images:
            if image.acquisition is None:
                raise ValueError(
                    "Cannot get acquisition paths for Zarr files without "
                    "'acquisition' metadata at the well level"
                )
            if image.acquisition in acquisition_dict:
                raise NotImplementedError(
                    "This task is not implemented for wells with multiple "
                    "images of the same acquisition"
                )
            acquisition_dict[image.acquisition] = image.path
        return acquisition_dict


def load_NgffImageMeta(zarr_path: str) -> NgffImageMeta:
    """
    Load the attributes of a zarr group and cast them to `NgffImageMeta`.

    Args:
        zarr_path: Path to the zarr group.

    Returns:
        A new `NgffImageMeta` object.
    """
    zarr_group = zarr.open_group(zarr_path, mode="r")
    zarr_attrs = zarr_group.attrs.asdict()
    # FIXME: add a try/except block. If it fails, the error should be made
    # informative.
    try:
        return NgffImageMeta(**zarr_attrs)
    except Exception as e:
        from devtools import debug  # FIXME remove

        debug(zarr_attrs)
        debug(e)
        raise e


def load_NgffWellMeta(zarr_path: str) -> NgffWellMeta:
    """
    Load the attributes of a zarr group and cast them to `NgffWellMeta`.

    Args:
        zarr_path: Path to the zarr group.

    Returns:
        A new `NgffWellMeta` object.
    """
    zarr_group = zarr.open_group(zarr_path, mode="r")
    zarr_attrs = zarr_group.attrs.asdict()
    # FIXME: add a try/except block. If it fails, the error should be made
    # informative.
    try:
        return NgffWellMeta(**zarr_attrs)
    except Exception as e:
        from devtools import debug  # FIXME remove

        debug(zarr_attrs)
        debug(e)
        raise e
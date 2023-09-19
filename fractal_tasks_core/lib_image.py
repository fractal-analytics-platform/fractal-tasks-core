import logging
from enum import Enum
from typing import Literal
from typing import Optional
from typing import Union

from pydantic import BaseModel
from pydantic import Field
from pydantic import validator


class Version(Enum):
    field_0_4 = "0.4"


class Window(BaseModel):
    end: float
    max: float
    min: float
    start: float


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
        if v.coordinateTransformations is not None:
            raise NotImplementedError(
                "Global coordinateTransformations at the multiscales "
                "level are not currently supported."
            )


class NgffImage(BaseModel):
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
    def axes(self) -> list[Axe]:
        return self.multiscale.axes

    @property
    def datasets(self) -> list[Dataset]:
        return self.multiscale.datasets

    @property
    def num_levels(self) -> int:
        return len(self.datasets)

    @property
    def pixel_sizes_zyx(self) -> list[tuple[float, float, float]]:
        # Construct pixel_sizes_zyx
        from devtools import debug

        debug(self.multiscale.axes)
        axes = [ax.name for ax in self.multiscale.axes]
        x_index = axes.index("x")
        y_index = axes.index("y")
        try:
            z_index = axes.index("z")
        except ValueError:
            z_index = None
            logging.warning(
                f"Z axis is not present ({axes=}), and Z pixel size is set"
                " to 1. This may work, by accident, but it is not fully"
                " supported."
            )
        pixel_sizes_zyx = []
        for level in range(self.num_levels):
            scale = self.datasets[level].coordinateTransformations[0].scale
            pixel_size_x = scale[x_index]
            pixel_size_y = scale[y_index]
            if z_index is not None:
                pixel_size_z = scale[z_index]
            else:
                pixel_size_z = 1.0
            pixel_sizes_zyx.append((pixel_size_z, pixel_size_y, pixel_size_x))
            pass
        return pixel_sizes_zyx

    def get_pixel_sizes_zyx(self, *, level: int) -> tuple[float, float, float]:
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
                    "is not supported."
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

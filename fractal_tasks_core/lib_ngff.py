"""
Pydantic models related to OME-NGFF 0.4 specs.
"""
import logging
from typing import Literal
from typing import Optional
from typing import Union

import zarr
from pydantic import BaseModel
from pydantic import Field
from pydantic import validator


class Window(BaseModel):
    """
    Model for `Channel.window`.

    Note that we deviate by NGFF specs by making `start` and `end` optional.
    See https://ngff.openmicroscopy.org/0.4/#omero-md.
    """

    max: float
    min: float
    start: Optional[float] = None
    end: Optional[float] = None


class Channel(BaseModel):
    """
    Model for an element of `Omero.channels`.

    See https://ngff.openmicroscopy.org/0.4/#omero-md.
    """

    window: Window
    label: Optional[str] = None
    family: Optional[str] = None
    color: str
    active: Optional[bool] = None


class Omero(BaseModel):
    """
    Model for `NgffImageMeta.omero`.

    See https://ngff.openmicroscopy.org/0.4/#omero-md.
    """

    channels: list[Channel]


class Axis(BaseModel):
    """
    Model for an element of `Multiscale.axes`.

    See https://ngff.openmicroscopy.org/0.4/#axes-md.
    """

    name: str
    type: Optional[str] = None


class ScaleCoordinateTransformation(BaseModel):
    """
    Model for a scale transformation.

    This corresponds to scale-type elements of
    `Dataset.coordinateTransformations` or
    `Multiscale.coordinateTransformations`.
    See https://ngff.openmicroscopy.org/0.4/#trafo-md
    """

    type: Literal["scale"]
    scale: list[float] = Field(..., min_items=2)


class TranslationCoordinateTransformation(BaseModel):
    """
    Model for a translation transformation.

    This corresponds to translation-type elements of
    `Dataset.coordinateTransformations` or
    `Multiscale.coordinateTransformations`.
    See https://ngff.openmicroscopy.org/0.4/#trafo-md
    """

    type: Literal["translation"]
    translation: list[float] = Field(..., min_items=2)


class Dataset(BaseModel):
    """
    Model for an element of `Multiscale.datasets`.

    See https://ngff.openmicroscopy.org/0.4/#multiscale-md
    """

    path: str
    coordinateTransformations: list[
        Union[
            ScaleCoordinateTransformation, TranslationCoordinateTransformation
        ]
    ] = Field(..., min_items=1)

    @property
    def scale_transformation(self) -> ScaleCoordinateTransformation:
        """
        Extract the unique scale transformation, or fail otherwise.
        """
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
    """
    Model for an elemeng of `NgffImageMeta.multiscales`.

    See https://ngff.openmicroscopy.org/0.4/#multiscale-md.
    """

    name: Optional[str] = None
    datasets: list[Dataset] = Field(..., min_items=1)
    version: Optional[str] = None
    axes: list[Axis] = Field(..., max_items=5, min_items=2, unique_items=True)
    coordinateTransformations: Optional[
        list[
            Union[
                ScaleCoordinateTransformation,
                TranslationCoordinateTransformation,
            ]
        ]
    ] = None

    @validator("coordinateTransformations", always=True)
    def _no_global_coordinateTransformations(cls, v):
        """
        Fail if Multiscale has a (global) coordinateTransformations attribute.
        """
        if v is not None:
            raise NotImplementedError(
                "Global coordinateTransformations at the multiscales "
                "level are not currently supported."
            )


class NgffImageMeta(BaseModel):
    """
    Model for the metadata of a NGFF image.

    See https://ngff.openmicroscopy.org/0.4/#image-layout.
    """

    multiscales: list[Multiscale] = Field(
        ...,
        description="The multiscale datasets for this image",
        min_items=1,
        unique_items=True,
    )
    omero: Optional[Omero] = None

    @property
    def multiscale(self) -> Multiscale:
        """
        The single element of `self.multiscales`.

        Raises:
            NotImplementedError:
                If there are no multiscales or more than one.
        """
        if len(self.multiscales) > 1:
            raise NotImplementedError(
                "Only images with one multiscale are supported "
                f"(given: {len(self.multiscales)}"
            )
        return self.multiscales[0]

    @property
    def datasets(self) -> list[Dataset]:
        """
        The `datasets` attribute of `self.multiscale`.
        """
        return self.multiscale.datasets

    @property
    def num_levels(self) -> int:
        return len(self.datasets)

    @property
    def axes_names(self) -> list[str]:
        """
        List of axes names.
        """
        return [ax.name for ax in self.multiscale.axes]

    @property
    def pixel_sizes_zyx(self) -> list[list[float]]:
        """
        Pixel sizes extracted from scale transformations of datasets.

        Raises:
            ValueError:
                If pixel sizes are below a given threshold (1e-9).
        """
        x_index = self.axes_names.index("x")
        y_index = self.axes_names.index("y")
        try:
            z_index = self.axes_names.index("z")
        except ValueError:
            z_index = None
            logging.warning(
                f"Z axis is not present (axes: {self.axes_names}), and Z pixel"
                " size is set to 1. This may work, by accident, but it is "
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
            _pixel_sizes_zyx.append([pixel_size_z, pixel_size_y, pixel_size_x])
            if min(_pixel_sizes_zyx[-1]) < 1e-9:
                raise ValueError(
                    f"Pixel sizes at level {level} are too small: "
                    f"{_pixel_sizes_zyx[-1]}"
                )

        return _pixel_sizes_zyx

    def get_pixel_sizes_zyx(self, *, level: int = 0) -> list[float]:
        return self.pixel_sizes_zyx[level]

    @property
    def coarsening_xy(self) -> int:
        """
        Linear coarsening factor in the YX plane.

        We only support coarsening factors that are homogeneous (both in the
        X/Y directions and across pyramid levels).

        Raises:
            NotImplementedError:
                If coarsening ratios are not homogeneous.
        """
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
    Model for an element of `Well.images`.

    See https://ngff.openmicroscopy.org/0.4/#well-md.

    NOTE: we deviate from NGFF specs, since we allow `path` to be an arbitrary
    string.
    TODO: include a check like `constr(regex=r'^[A-Za-z0-9]+$')`, through a
    Pydantic validator.
    """

    acquisition: Optional[int] = Field(
        None, description="A unique identifier within the context of the plate"
    )
    path: str = Field(
        ..., description="The path for this field of view subgroup"
    )


class Well(BaseModel):
    """
    Model for `NgffWellMeta.well`.

    See https://ngff.openmicroscopy.org/0.4/#well-md.
    """

    images: list[Image] = Field(
        ...,
        description="The images included in this well",
        min_items=1,
        unique_items=True,
    )
    version: Optional[str] = Field(
        None, description="The version of the specification"
    )


class NgffWellMeta(BaseModel):
    """
    Model for the metadata of a NGFF well.

    See https://ngff.openmicroscopy.org/0.4/#well-md.
    """

    well: Optional[Well] = None

    def get_acquisition_paths(self) -> dict[int, str]:
        """
        Create mapping from acquisition indices to corresponding paths.

        Runs on the well zarr attributes and loads the relative paths in the
        well.

        Returns:
            Dictionary with `(acquisition index: image path)` key/value pairs.

        Raises:
            ValueError:
                If an element of `self.well.images` has no `acquisition`
                    attribute.
            NotImplementedError:
                If acquisitions are not unique.
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
    zarr_group = zarr.open_group(zarr_path, mode="r")
    zarr_attrs = zarr_group.attrs.asdict()
    try:
        return NgffWellMeta(**zarr_attrs)
    except Exception as e:
        logging.error(
            f"Contents of {zarr_path} cannot be cast to NgffWellMeta.\n"
            f"Original error:\n{str(e)}"
        )
        raise e

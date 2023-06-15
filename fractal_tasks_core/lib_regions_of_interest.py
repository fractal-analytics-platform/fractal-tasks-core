"""
Copyright 2022 (C)
    Friedrich Miescher Institute for Biomedical Research and
    University of Zurich

    Original authors:
    Tommaso Comparin <tommaso.comparin@exact-lab.it>
    Joel Lüthi <joel.luethi@uzh.ch>

    This file is part of Fractal and was originally developed by eXact lab
    S.r.l.  <exact-lab.it> under contract with Liberali Lab from the Friedrich
    Miescher Institute for Biomedical Research and Pelkmans Lab from the
    University of Zurich.

Functions to handle regions of interests (via pandas and AnnData)
"""
import logging
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import anndata as ad
import dask.array as da
import numpy as np
import pandas as pd
import zarr


logger = logging.getLogger(__name__)


def prepare_FOV_ROI_table(
    df: pd.DataFrame, metadata: list[str] = ["time"]
) -> ad.AnnData:
    """
    Description

    :param dummy: this is just a placeholder
    :type dummy: int
    """

    # Make a local copy of the dataframe, to avoid SettingWithCopyWarning
    df = df.copy()

    # Convert DataFrame index to str, to avoid
    # >> ImplicitModificationWarning: Transforming to str index
    # when creating AnnData object.
    # Do this in the beginning to allow concatenation with e.g. time
    df.index = df.index.astype(str)

    # Obtain box size in physical units
    df = df.assign(len_x_micrometer=df.x_pixel * df.pixel_size_x)
    df = df.assign(len_y_micrometer=df.y_pixel * df.pixel_size_y)
    df = df.assign(len_z_micrometer=df.z_pixel * df.pixel_size_z)

    # Select only the numeric positional columns needed to define ROIs
    # (to avoid) casting things like the data column to float32
    # or to use unnecessary columns like bit_depth
    positional_columns = [
        "x_micrometer",
        "y_micrometer",
        "z_micrometer",
        "len_x_micrometer",
        "len_y_micrometer",
        "len_z_micrometer",
        "x_micrometer_original",
        "y_micrometer_original",
    ]

    # Assign dtype explicitly, to avoid
    # >> UserWarning: X converted to numpy array with dtype float64
    # when creating AnnData object
    df_roi = df.loc[:, positional_columns].astype(np.float32)

    # Create an AnnData object directly from the DataFrame
    adata = ad.AnnData(X=df_roi)

    # Save any metadata that is specified to the obs df
    for col in metadata:
        if col in df:
            # Cast all metadata to str.
            # Reason: AnnData Zarr writers don't support all pandas types.
            # e.g. pandas.core.arrays.datetimes.DatetimeArray can't be written
            adata.obs[col] = df[col].astype(str)

    # Rename rows and columns: Maintain FOV indices from the dataframe
    # (they are already enforced to be unique by Pandas and may contain
    # information for the user, as they are based on the filenames)
    adata.obs_names = "FOV_" + adata.obs.index
    adata.var_names = list(map(str, df_roi.columns))

    return adata


def prepare_well_ROI_table(
    df: pd.DataFrame, metadata: list[str] = ["time"]
) -> ad.AnnData:
    """
    Description

    :param dummy: this is just a placeholder
    :type dummy: int
    """

    # Make a local copy of the dataframe, to avoid SettingWithCopyWarning
    df = df.copy()

    # Convert DataFrame index to str, to avoid
    # >> ImplicitModificationWarning: Transforming to str index
    # when creating AnnData object.
    # Do this in the beginning to allow concatenation with e.g. time
    df.index = df.index.astype(str)

    # Calculate bounding box extents in physical units
    for mu in ["x", "y", "z"]:
        # Obtain per-FOV properties in physical units.
        # NOTE: a FOV ROI is defined here as the interval [min_micrometer,
        # max_micrometer], with max_micrometer=min_micrometer+len_micrometer
        min_micrometer = df[f"{mu}_micrometer"]
        len_micrometer = df[f"{mu}_pixel"] * df[f"pixel_size_{mu}"]
        max_micrometer = min_micrometer + len_micrometer
        # Obtain well bounding box, in physical units
        min_min_micrometer = min_micrometer.min()
        max_max_micrometer = max_micrometer.max()
        df[f"{mu}_micrometer"] = min_min_micrometer
        df[f"len_{mu}_micrometer"] = max_max_micrometer - min_min_micrometer

    # Select only the numeric positional columns needed to define ROIs
    # (to avoid) casting things like the data column to float32
    # or to use unnecessary columns like bit_depth
    positional_columns = [
        "x_micrometer",
        "y_micrometer",
        "z_micrometer",
        "len_x_micrometer",
        "len_y_micrometer",
        "len_z_micrometer",
    ]

    # Assign dtype explicitly, to avoid
    # >> UserWarning: X converted to numpy array with dtype float64
    # when creating AnnData object
    df_roi = df.iloc[0:1, :].loc[:, positional_columns].astype(np.float32)

    # Create an AnnData object directly from the DataFrame
    adata = ad.AnnData(X=df_roi)

    # Save any metadata that is specified to the obs df
    for col in metadata:
        if col in df:
            # Cast all metadata to str.
            # Reason: AnnData Zarr writers don't support all pandas types.
            # e.g. pandas.core.arrays.datetimes.DatetimeArray can't be written
            adata.obs[col] = df[col].astype(str)

    # Rename rows and columns: Maintain FOV indices from the dataframe
    # (they are already enforced to be unique by Pandas and may contain
    # information for the user, as they are based on the filenames)
    adata.obs_names = "well_" + adata.obs.index
    adata.var_names = list(map(str, df_roi.columns))

    return adata


def convert_ROIs_from_3D_to_2D(
    adata: ad.AnnData,
    pixel_size_z: float,
) -> ad.AnnData:
    """
    Description

    :param dummy: this is just a placeholder
    :type dummy: int
    """

    # Compress a 3D stack of images to a single Z plane,
    # with thickness equal to pixel_size_z
    df = adata.to_df()
    df["len_z_micrometer"] = pixel_size_z

    # Assign dtype explicitly, to avoid
    # >> UserWarning: X converted to numpy array with dtype float64
    # when creating AnnData object
    df = df.astype(np.float32)

    # Create an AnnData object directly from the DataFrame
    new_adata = ad.AnnData(X=df)

    # Rename rows and columns
    new_adata.obs_names = adata.obs_names
    new_adata.var_names = list(map(str, df.columns))

    return new_adata


def convert_ROI_table_to_indices(
    ROI: ad.AnnData,
    full_res_pxl_sizes_zyx: Sequence[float],
    level: int = 0,
    coarsening_xy: int = 2,
    cols_xyz_pos: Sequence[str] = [
        "x_micrometer",
        "y_micrometer",
        "z_micrometer",
    ],
    cols_xyz_len: Sequence[str] = [
        "len_x_micrometer",
        "len_y_micrometer",
        "len_z_micrometer",
    ],
    reset_origin: bool = True,
) -> list[list[int]]:
    """
    Description

    FIXME add docstring

    :param dummy: this is just a placeholder
    :type dummy: int
    """
    # Handle empty ROI table
    if len(ROI) == 0:
        return []

    # Set pyramid-level pixel sizes
    pxl_size_z, pxl_size_y, pxl_size_x = full_res_pxl_sizes_zyx
    prefactor = coarsening_xy**level
    pxl_size_x *= prefactor
    pxl_size_y *= prefactor

    x_pos, y_pos, z_pos = cols_xyz_pos[:]
    x_len, y_len, z_len = cols_xyz_len[:]

    # FIXME: see discussion on ROI-table origin at
    # https://github.com/fractal-analytics-platform/fractal-tasks-core/issues/339
    if reset_origin:
        origin_x = min(ROI[:, x_pos].X[:, 0])
        origin_y = min(ROI[:, y_pos].X[:, 0])
        origin_z = min(ROI[:, z_pos].X[:, 0])
    else:
        origin_x = 0.0
        origin_y = 0.0
        origin_z = 0.0

    list_indices = []
    for FOV in ROI.obs_names:

        # Extract data from anndata table
        x_micrometer = ROI[FOV, x_pos].X[0, 0] - origin_x
        y_micrometer = ROI[FOV, y_pos].X[0, 0] - origin_y
        z_micrometer = ROI[FOV, z_pos].X[0, 0] - origin_z
        len_x_micrometer = ROI[FOV, x_len].X[0, 0]
        len_y_micrometer = ROI[FOV, y_len].X[0, 0]
        len_z_micrometer = ROI[FOV, z_len].X[0, 0]

        # Identify indices along the three dimensions
        start_x = x_micrometer / pxl_size_x
        end_x = (x_micrometer + len_x_micrometer) / pxl_size_x
        start_y = y_micrometer / pxl_size_y
        end_y = (y_micrometer + len_y_micrometer) / pxl_size_y
        start_z = z_micrometer / pxl_size_z
        end_z = (z_micrometer + len_z_micrometer) / pxl_size_z
        indices = [start_z, end_z, start_y, end_y, start_x, end_x]

        # Round indices to lower integer
        indices = list(map(round, indices))

        # Append ROI indices to to list
        list_indices.append(indices[:])

    return list_indices


def _inspect_ROI_table(
    path: str,
    full_res_pxl_sizes_zyx: Sequence[float],
    level: int = 0,
    coarsening_xy: int = 2,
) -> None:
    """
    Description

    :param dummy: this is just a placeholder
    :type dummy: int
    """

    print(f"{full_res_pxl_sizes_zyx=}")

    adata = ad.read_zarr(path)
    df = adata.to_df()
    print("table")
    print(df)
    print()

    try:
        list_indices = convert_ROI_table_to_indices(
            adata,
            level=level,
            coarsening_xy=coarsening_xy,
            full_res_pxl_sizes_zyx=full_res_pxl_sizes_zyx,
            # verbose=True,
        )
        print()
        print(f"level:         {level}")
        print(f"coarsening_xy: {coarsening_xy}")
        print("list_indices:")
        for indices in list_indices:
            print(indices)
        print()
    except KeyError as e:
        print("Something went wrong in convert_ROI_table_to_indices\n", str(e))

    return df


def array_to_bounding_box_table(
    mask_array: np.ndarray, pxl_sizes_zyx: list[float]
) -> pd.DataFrame:

    """
    Description

    :param dummy: this is just a placeholder
    :type dummy: int
    """

    labels = np.unique(mask_array)
    labels = labels[labels > 0]
    elem_list = []
    for label in labels:
        label_match = np.where(mask_array == label)
        # FIXME: multiplication of np.ndarray with list
        zmin, ymin, xmin = np.min(label_match, axis=1) * pxl_sizes_zyx
        zmax, ymax, xmax = (np.max(label_match, axis=1) + 1) * pxl_sizes_zyx

        length_x = xmax - xmin
        length_y = ymax - ymin
        length_z = zmax - zmin
        elem_list.append((xmin, ymin, zmin, length_x, length_y, length_z))

    df_columns = [
        "x_micrometer",
        "y_micrometer",
        "z_micrometer",
        "len_x_micrometer",
        "len_y_micrometer",
        "len_z_micrometer",
    ]

    if len(elem_list) == 0:
        df = pd.DataFrame(columns=[x for x in df_columns] + ["label"])
    else:
        df = pd.DataFrame(np.array(elem_list), columns=df_columns)
        df["label"] = labels

    return df


def is_ROI_table_valid(*, table_path: str, use_masks: bool) -> Optional[bool]:
    """
    Verify some validity assumptions on a ROI table

    This function reflects our current working assumptions (e.g. the presence
    of some specific columns); this may change in future versions.

    If ``use_masks=True``, we verify that the table is suitable to be used as
    part of our masked-loading functions (see ``lib_masked_loading.py``); if
    these checks fail, ``use_masks`` should be set to ``False`` upstream in the
    parent function.

    :param table_path: Path of the AnnData ROI table to be checked.
    :param use_masks: If ``True``, perform some additional checks related to
                      masked loading.
    :returns: Always ``None`` if ``use_masks=False``, otherwise return whether
             the table is valid for masked loading.
    """

    # Hard constraint: table columns must include some expected ones
    table = ad.read_zarr(table_path)
    columns = [
        "x_micrometer",
        "y_micrometer",
        "z_micrometer",
        "len_x_micrometer",
        "len_y_micrometer",
        "len_z_micrometer",
    ]
    for column in columns:
        if column not in table.var_names:
            raise ValueError(f"Column {column} is not present in ROI table")
    if not use_masks:
        return None

    # Soft constraint: the table can be used for masked loading (if not, return
    # False)
    attrs = zarr.group(table_path).attrs
    logger.info(f"ROI table at {table_path} has attrs: {attrs.asdict()}")
    valid = set(("type", "region", "instance_key")).issubset(attrs.keys())
    if valid:
        valid = valid and attrs["type"] == "ngff:region_table"
        valid = valid and "path" in attrs["region"].keys()
    if valid:
        return True
    else:
        return False


def load_region(
    data_zyx: da.Array,
    region: Tuple[slice, slice, slice],
    compute=True,
    return_as_3D=False,
) -> Union[da.Array, np.ndarray]:
    """
    Load a region from a dask array

    Can handle both 2D and 3D dask arrays as input and return them as is or
    always as a 3D array

    :param data_zyx: dask array, 2D or 3D
    :param region: region to load, tuple of three slices (ZYX)
    :param compute: whether to compute the result. If True, returns a numpy
                    array. If False, returns a dask array.
    :return_as_3D: whether to return a 3D array, even if the input is 2D
    :return: 3D array
    """

    if len(region) != 3:
        raise ValueError(
            f"In `load_region`, `region` must have three elements "
            f"(given: {len(region)})."
        )

    if len(data_zyx.shape) == 3:
        img = data_zyx[region]
    elif len(data_zyx.shape) == 2:
        img = data_zyx[(region[1], region[2])]
        if return_as_3D:
            img = np.expand_dims(img, axis=0)
    else:
        raise ValueError(
            f"Shape {data_zyx.shape} not supported for `load_region`"
        )
    if compute:
        return img.compute()
    else:
        return img

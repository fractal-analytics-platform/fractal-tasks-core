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
from typing import List
from typing import Sequence

import anndata as ad
import numpy as np
import pandas as pd


def prepare_FOV_ROI_table(
    df: pd.DataFrame, metadata: list = ["time"]
) -> ad.AnnData:
    """
    Description

    :param dummy: this is just a placeholder
    :type dummy: int
    """

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
    df: pd.DataFrame, metadata: list = ["time"]
) -> ad.AnnData:
    """
    Description

    :param dummy: this is just a placeholder
    :type dummy: int
    """

    # Convert DataFrame index to str, to avoid
    # >> ImplicitModificationWarning: Transforming to str index
    # when creating AnnData object.
    # Do this in the beginning to allow concatenation with e.g. time
    df.index = df.index.astype(str)

    # Calculate bounding box extents in physical units
    min_micrometers = {}
    max_micrometers = {}
    for mu in ["x", "y", "z"]:
        # Reset reference values for coordinates
        df[f"{mu}_micrometer"] -= df[f"{mu}_micrometer"].min()
        # Obtain FOV box size in physical units
        df[f"len_{mu}_micrometer"] = df[f"{mu}_pixel"] * df[f"pixel_size_{mu}"]
        # Obtain well bounding box, in physical units
        min_micrometers[mu] = df[f"{mu}_micrometer"].min()
        max_micrometers[mu] = (
            df[f"{mu}_micrometer"] + df[f"len_{mu}_micrometer"]
        ).max()
        df[f"len_{mu}_micrometer"] = max_micrometers[mu]

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
    adata: ad.AnnData = None, pixel_size_z: float = None
) -> ad.AnnData:
    """
    Description

    :param dummy: this is just a placeholder
    :type dummy: int
    """

    if pixel_size_z is None:
        raise Exception("Missing pixel_size_z in convert_ROIs_from_3D_to_2D")

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
    level: int = 0,
    coarsening_xy: int = 2,
    full_res_pxl_sizes_zyx: Sequence[float] = None,
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
) -> List[List[int]]:
    """
    Description

    :param dummy: this is just a placeholder
    :type dummy: int
    """

    # Set pyramid-level pixel sizes
    pxl_size_z, pxl_size_y, pxl_size_x = full_res_pxl_sizes_zyx
    prefactor = coarsening_xy**level
    pxl_size_x *= prefactor
    pxl_size_y *= prefactor

    x_pos, y_pos, z_pos = cols_xyz_pos[:]
    x_len, y_len, z_len = cols_xyz_len[:]

    origin_x = min(ROI[:, x_pos].X[:, 0])
    origin_y = min(ROI[:, y_pos].X[:, 0])
    origin_z = min(ROI[:, z_pos].X[:, 0])

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
    path: str = None,
    level: int = 0,
    coarsening_xy: int = 2,
    full_res_pxl_sizes_zyx=[1.0, 0.1625, 0.1625],
) -> None:
    """
    Description

    :param dummy: this is just a placeholder
    :type dummy: int
    """

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
        )

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
    mask_array: np.ndarray, pxl_sizes_zyx: List[float]
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

    ann_df = pd.DataFrame(np.array(elem_list), columns=df_columns)

    return ann_df

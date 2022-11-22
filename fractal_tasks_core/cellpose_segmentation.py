"""
Copyright 2022 (C)
    Friedrich Miescher Institute for Biomedical Research and
    University of Zurich

    Original authors:
    Tommaso Comparin <tommaso.comparin@exact-lab.it>
    Marco Franzon <marco.franzon@exact-lab.it>
    Joel Lüthi  <joel.luethi@fmi.ch>

    This file is part of Fractal and was originally developed by eXact lab
    S.r.l.  <exact-lab.it> under contract with Liberali Lab from the Friedrich
    Miescher Institute for Biomedical Research and Pelkmans Lab from the
    University of Zurich.

Image segmentation via cellpose library
"""
import json
import logging
import os
import time
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Literal
from typing import Optional
from typing import Sequence

import anndata as ad
import dask.array as da
import numpy as np
import pandas as pd
import zarr
from anndata.experimental import write_elem
from cellpose import models
from cellpose.core import use_gpu

import fractal_tasks_core
from fractal_tasks_core.lib_pyramid_creation import build_pyramid
from fractal_tasks_core.lib_regions_of_interest import (
    array_to_bounding_box_table,
)
from fractal_tasks_core.lib_regions_of_interest import (
    convert_ROI_table_to_indices,
)
from fractal_tasks_core.lib_remove_FOV_overlaps import (
    get_overlapping_pairs_3D,
)
from fractal_tasks_core.lib_zattrs_utils import extract_zyx_pixel_sizes
from fractal_tasks_core.lib_zattrs_utils import rescale_datasets

logger = logging.getLogger(__name__)


__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__


def segment_FOV(
    column,
    model=None,
    do_3D=True,
    anisotropy=None,
    diameter=40.0,
    cellprob_threshold=0.0,
    flow_threshold=0.4,
    label_dtype=None,
    well_id=None,
):
    """
    Description

    :param dummy: this is a placeholder
    :param dummy: int
    """

    # Write some debugging info
    logger.info(
        f"[{well_id}][segment_FOV] START Cellpose |"
        f" column: {type(column)}, {column.shape} |"
        f" do_3D: {do_3D} |"
        f" model.diam_mean: {model.diam_mean} |"
        f" diameter: {diameter} |"
        f" flow threshold: {flow_threshold}"
    )

    # Actual labeling
    t0 = time.perf_counter()
    mask, flows, styles = model.eval(
        column,
        channels=[0, 0],
        do_3D=do_3D,
        net_avg=False,
        augment=False,
        diameter=diameter,
        anisotropy=anisotropy,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
    )
    if not do_3D:
        mask = np.expand_dims(mask, axis=0)
    t1 = time.perf_counter()

    # Write some debugging info
    logger.info(
        f"[{well_id}][segment_FOV] END   Cellpose |"
        f" Elapsed: {t1-t0:.4f} seconds |"
        f" mask shape: {mask.shape},"
        f" mask dtype: {mask.dtype} (before recast to {label_dtype}),"
        f" max(mask): {np.max(mask)} |"
        f" model.diam_mean: {model.diam_mean} |"
        f" diameter: {diameter} |"
        f" flow threshold: {flow_threshold}"
    )

    return mask.astype(label_dtype)


def cellpose_segmentation(
    *,
    # Fractal arguments
    input_paths: Sequence[Path],
    output_path: Path,
    component: str,
    metadata: Dict[str, Any],
    # Task-specific arguments
    labeling_channel: str,
    labeling_level: int = 1,
    relabeling: bool = True,
    anisotropy: Optional[float] = None,
    diameter_level0: float = 80.0,
    cellprob_threshold: float = 0.0,
    flow_threshold: float = 0.4,
    ROI_table_name: str = "FOV_ROI_table",
    bounding_box_ROI_table_name: Optional[str] = None,
    label_name: Optional[str] = None,
    model_type: Literal["nuclei", "cyto", "cyto2"] = "nuclei",
    pretrained_model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Example inputs:
      input_paths: PosixPath('tmp_out_mip/*.zarr')
      output_path: PosixPath('tmp_out_mip/*.zarr')
      component: myplate.zarr/B/03/0/
      metadata: {...}

    :param input_paths: TBD (fractal default arg)
    :param output_path: TBD (fractal default arg)
    :param metadata: TBD (fractal default arg)
    :param component: TBD (fractal default arg)
    :param labeling_channel: TBD
    :param labeling_level: TBD
    :param relabeling: TBD
    :param anisotropy: TBD
    :param diameter_level0: TBD
    :param cellprob_threshold: TBD
    :param flow_threshold: TBD
    :param ROI_table_name: TBD
    :param bounding_box_ROI_table_name: TBD
    :param label_name: TBD
    :param model_type: TBD
    :param pretrained_model: TBD. If not ``None``, this takes precedence
                             over ``model_type``.
    """

    # Set input path
    if len(input_paths) > 1:
        raise NotImplementedError
    in_path = input_paths[0].parent
    zarrurl = (in_path.resolve() / component).as_posix() + "/"
    logger.info(zarrurl)

    # Read useful parameters from metadata
    num_levels = metadata["num_levels"]
    coarsening_xy = metadata["coarsening_xy"]
    chl_list = metadata["channel_list"]
    plate, well = component.split(".zarr/")

    # Find well ID
    well_id = well.replace("/", "_")[:-1]

    # Find channel index
    if labeling_channel not in chl_list:
        raise Exception(f"ERROR: {labeling_channel} not in {chl_list}")
    ind_channel = chl_list.index(labeling_channel)

    # Load ZYX data
    data_zyx = da.from_zarr(f"{zarrurl}{labeling_level}")[ind_channel]
    logger.info(f"[{well_id}] {data_zyx.shape=}")

    # Read ROI table
    ROI_table = ad.read_zarr(f"{zarrurl}tables/{ROI_table_name}")

    # Read pixel sizes from zattrs file
    full_res_pxl_sizes_zyx = extract_zyx_pixel_sizes(
        f"{zarrurl}.zattrs", level=0
    )

    actual_res_pxl_sizes_zyx = extract_zyx_pixel_sizes(
        f"{zarrurl}.zattrs", level=labeling_level
    )
    # Create list of indices for 3D FOVs spanning the entire Z direction
    list_indices = convert_ROI_table_to_indices(
        ROI_table,
        level=labeling_level,
        coarsening_xy=coarsening_xy,
        full_res_pxl_sizes_zyx=full_res_pxl_sizes_zyx,
    )

    # Extract image size from FOV-ROI indices
    # Note: this works at level=0, where FOVs should all be of the exact same
    #       size (in pixels)
    FOV_ROI_table = ad.read_zarr(f"{zarrurl}tables/FOV_ROI_table")
    list_FOV_indices_level0 = convert_ROI_table_to_indices(
        FOV_ROI_table,
        level=0,
        full_res_pxl_sizes_zyx=full_res_pxl_sizes_zyx,
    )
    ref_img_size = None
    for indices in list_FOV_indices_level0:
        img_size = (indices[3] - indices[2], indices[5] - indices[4])
        if ref_img_size is None:
            ref_img_size = img_size
        else:
            if img_size != ref_img_size:
                raise Exception(
                    "ERROR: inconsistent image sizes in "
                    f"{list_FOV_indices_level0=}"
                )
    img_size_y, img_size_x = img_size[:]

    # Select 2D/3D behavior and set some parameters
    do_3D = data_zyx.shape[0] > 1
    if do_3D:
        if anisotropy is None:
            # Read pixel sizes from zattrs file
            pxl_zyx = extract_zyx_pixel_sizes(
                zarrurl + ".zattrs", level=labeling_level
            )
            pixel_size_z, pixel_size_y, pixel_size_x = pxl_zyx[:]
            logger.info(f"[{well_id}] {pxl_zyx=}")
            if not np.allclose(pixel_size_x, pixel_size_y):
                raise Exception(
                    "ERROR: XY anisotropy detected"
                    f"pixel_size_x={pixel_size_x}"
                    f"pixel_size_y={pixel_size_y}"
                )
            anisotropy = pixel_size_z / pixel_size_x

    # Prelminary checks on Cellpose model
    if pretrained_model is None:
        if model_type not in ["nuclei", "cyto2", "cyto"]:
            raise ValueError(f"ERROR model_type={model_type} is not allowed.")
    else:
        if not os.path.exists(pretrained_model):
            raise ValueError(f"{pretrained_model=} does not exist.")

    # Load zattrs file
    zattrs_file = f"{zarrurl}.zattrs"
    with open(zattrs_file, "r") as jsonfile:
        zattrs = json.load(jsonfile)

    # Preliminary checks on multiscales
    multiscales = zattrs["multiscales"]
    if len(multiscales) > 1:
        raise NotImplementedError(
            f"Found {len(multiscales)} multiscales, "
            "but only one is currently supported."
        )
    if "coordinateTransformations" in multiscales[0].keys():
        raise NotImplementedError(
            "global coordinateTransformations at the multiscales "
            "level are not currently supported"
        )

    # Set channel label
    if label_name is None:
        try:
            omero_label = zattrs["omero"]["channels"][ind_channel]["label"]
            label_name = f"label_{omero_label}"
        except (KeyError, IndexError):
            label_name = f"label_{ind_channel}"

    # Rescale datasets (only relevant for labeling_level>0)
    new_datasets = rescale_datasets(
        datasets=multiscales[0]["datasets"],
        coarsening_xy=coarsening_xy,
        reference_level=labeling_level,
    )

    # Write zattrs for labels and for specific label
    # FIXME deal with: (1) many channels, (2) overwriting
    labels_group = zarr.group(f"{zarrurl}labels")
    labels_group.attrs["labels"] = [label_name]
    label_group = labels_group.create_group(label_name)
    label_group.attrs["image-label"] = {"version": __OME_NGFF_VERSION__}
    label_group.attrs["multiscales"] = [
        {
            "name": label_name,
            "version": __OME_NGFF_VERSION__,
            "axes": [
                ax for ax in multiscales[0]["axes"] if ax["type"] != "channel"
            ],
            "datasets": new_datasets,
        }
    ]

    # Open new zarr group for mask 0-th level
    logger.info(f"[{well_id}] {zarrurl}labels/{label_name}/0")
    zarr.group(f"{zarrurl}/labels")
    zarr.group(f"{zarrurl}/labels/{label_name}")
    store = da.core.get_mapper(f"{zarrurl}labels/{label_name}/0")
    label_dtype = np.uint32
    mask_zarr = zarr.create(
        shape=data_zyx.shape,
        chunks=data_zyx.chunksize,
        dtype=label_dtype,
        store=store,
        overwrite=False,
        dimension_separator="/",
    )

    logger.info(
        f"[{well_id}] "
        f"mask will have shape {data_zyx.shape} "
        f"and chunks {data_zyx.chunks}"
    )

    # Initialize cellpose
    gpu = use_gpu()
    if pretrained_model:
        model = models.CellposeModel(
            gpu=gpu, pretrained_model=pretrained_model
        )
    else:
        model = models.CellposeModel(gpu=gpu, model_type=model_type)

    # Initialize other things
    logger.info(f"[{well_id}] Start cellpose_segmentation task for {zarrurl}")
    logger.info(f"[{well_id}] relabeling: {relabeling}")
    logger.info(f"[{well_id}] do_3D: {do_3D}")
    logger.info(f"[{well_id}] use_gpu: {gpu}")
    logger.info(f"[{well_id}] labeling_level: {labeling_level}")
    logger.info(f"[{well_id}] model_type: {model_type}")
    logger.info(f"[{well_id}] pretrained_model: {pretrained_model}")
    logger.info(f"[{well_id}] anisotropy: {anisotropy}")
    logger.info(f"[{well_id}] Total well shape/chunks:")
    logger.info(f"[{well_id}] {data_zyx.shape}")
    logger.info(f"[{well_id}] {data_zyx.chunks}")

    # Counters for relabeling
    if relabeling:
        num_labels_tot = 0

    # Iterate over ROIs
    num_ROIs = len(list_indices)

    if bounding_box_ROI_table_name:
        bbox_dataframe_list = []

    logger.info(f"[{well_id}] Now starting loop over {num_ROIs} ROIs")
    for i_ROI, indices in enumerate(list_indices):
        # Define region
        s_z, e_z, s_y, e_y, s_x, e_x = indices[:]
        region = (
            slice(s_z, e_z),
            slice(s_y, e_y),
            slice(s_x, e_x),
        )
        logger.info(f"[{well_id}] Now processing ROI {i_ROI+1}/{num_ROIs}")
        # Execute illumination correction
        fov_mask = segment_FOV(
            data_zyx[s_z:e_z, s_y:e_y, s_x:e_x].compute(),
            model=model,
            do_3D=do_3D,
            anisotropy=anisotropy,
            label_dtype=label_dtype,
            diameter=diameter_level0 / coarsening_xy**labeling_level,
            cellprob_threshold=cellprob_threshold,
            flow_threshold=flow_threshold,
            well_id=well_id,
        )

        # Shift labels and update relabeling counters
        if relabeling:
            num_labels_fov = np.max(fov_mask)
            fov_mask[fov_mask > 0] += num_labels_tot
            num_labels_tot += num_labels_fov

            # Write some logs
            logger.info(
                f"[{well_id}] "
                f"FOV ROI {indices}, "
                f"{num_labels_fov=}, "
                f"{num_labels_tot=}"
            )

            # Check that total number of labels is under control
            if num_labels_tot > np.iinfo(label_dtype).max:
                raise Exception(
                    "ERROR in re-labeling:"
                    f"Reached {num_labels_tot} labels, "
                    f"but dtype={label_dtype}"
                )

        if bounding_box_ROI_table_name:

            bbox_df = array_to_bounding_box_table(
                fov_mask, actual_res_pxl_sizes_zyx
            )

            bbox_dataframe_list.append(bbox_df)

            overlap_list = []
            for df in bbox_dataframe_list:
                overlap_list.append(
                    get_overlapping_pairs_3D(df, full_res_pxl_sizes_zyx)
                )

        # Compute and store 0-th level to disk
        da.array(fov_mask).to_zarr(
            url=mask_zarr,
            region=region,
            compute=True,
        )

    logger.info(
        f"[{well_id}] End cellpose_segmentation task for {zarrurl}, "
        "now building pyramids."
    )

    # Starting from on-disk highest-resolution data, build and write to disk a
    # pyramid of coarser levels
    build_pyramid(
        zarrurl=f"{zarrurl}labels/{label_name}",
        overwrite=False,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        chunksize=data_zyx.chunksize,
        aggregation_function=np.max,
    )

    logger.info(f"[{well_id}] End building pyramids, exit")

    if bounding_box_ROI_table_name:
        logger.info(f"[{well_id}] Writing bounding box table, exit")
        # Concatenate all FOV dataframes
        df_well = pd.concat(bbox_dataframe_list, axis=0, ignore_index=True)
        df_well.index = df_well.index.astype(str)
        # Convert all to float (warning: some would be int, in principle)
        bbox_dtype = np.float32
        df_well = df_well.astype(bbox_dtype)
        # Convert to anndata
        bbox_table = ad.AnnData(df_well, dtype=bbox_dtype)
        # Write to zarr group
        group_tables = zarr.group(f"{in_path}/{component}/tables/")
        write_elem(group_tables, bounding_box_ROI_table_name, bbox_table)
        logger.info(
            f"[{in_path}/{component}/tables/{bounding_box_ROI_table_name}"
        )

    return {}


if __name__ == "__main__":

    from pydantic import BaseModel
    from fractal_tasks_core._utils import run_fractal_task

    class TaskArguments(BaseModel):
        # Fractal arguments
        input_paths: Sequence[Path]
        output_path: Path
        component: str
        metadata: Dict[str, Any]
        # Task-specific arguments
        labeling_channel: str
        labeling_level: int = 1
        relabeling: bool = True
        anisotropy: Optional[float] = None
        diameter_level0: float = 80.0
        cellprob_threshold: float = 0.0
        flow_threshold: float = 0.4
        ROI_table_name: str = "FOV_ROI_table"
        bounding_box_ROI_table_name: Optional[str] = None
        label_name: Optional[str] = None
        model_type: Literal["nuclei", "cyto", "cyto2"] = "nuclei"
        pretrained_model: Optional[str] = None

    run_fractal_task(
        task_function=cellpose_segmentation,
        TaskArgsModel=TaskArguments,
        logger_name=logger.name,
    )

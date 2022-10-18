import json
import logging
import os
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Iterable
from typing import Optional

import anndata as ad
import dask.array as da
import napari_workflows
import numpy as np
import pandas as pd
import zarr
from anndata.experimental import write_elem
from napari_workflows._io_yaml_v1 import load_workflow

import fractal_tasks_core
from .lib_pyramid_creation import build_pyramid
from .lib_regions_of_interest import convert_ROI_table_to_indices
from .lib_upscale_array import upscale_array
from .lib_zattrs_utils import extract_zyx_pixel_sizes
from .lib_zattrs_utils import rescale_datasets

__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__


logger = logging.getLogger(__name__)


def napari_workflows_wrapper(
    *,
    input_paths: Iterable[Path],
    output_path: Path,
    metadata: Optional[Dict[str, Any]] = None,
    component: str = None,
    input_specs: Dict[str, Dict[str, str]] = None,
    output_specs: Dict[str, Dict[str, str]] = None,
    workflow_file: str = None,
    ROI_table_name: str = "FOV_ROI_table",
):

    # Pre-processing of task inputs
    if len(input_paths) > 1:
        raise NotImplementedError("We currently only support a single in_path")
    in_path = input_paths[0].parent.as_posix()
    num_levels = metadata["num_levels"]
    coarsening_xy = metadata["coarsening_xy"]
    chl_list = metadata["channel_list"]
    label_dtype = np.uint32

    # Hard-coded parameters
    level = 0  # FIXME
    labeling_level = 0  # FIXME
    if level > 0 or labeling_level > 0:
        raise NotImplementedError(f"{level=}, {labeling_level=}")

    # Load zattrs file and multiscales
    zattrs_file = f"{in_path}/{component}/.zattrs"
    with open(zattrs_file, "r") as jsonfile:
        zattrs = json.load(jsonfile)
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

    # Read ROI table
    zarrurl = f"{in_path}/{component}"
    ROI_table = ad.read_zarr(f"{in_path}/{component}/tables/{ROI_table_name}")

    # Read pixel sizes from zattrs file
    full_res_pxl_sizes_zyx = extract_zyx_pixel_sizes(zattrs_file, level=0)

    # Create list of indices for 3D FOVs spanning the entire Z direction
    list_indices = convert_ROI_table_to_indices(
        ROI_table,
        level=labeling_level,
        coarsening_xy=coarsening_xy,
        full_res_pxl_sizes_zyx=full_res_pxl_sizes_zyx,
    )
    num_ROIs = len(list_indices)
    logger.info(
        f"Completed reading ROI table {ROI_table_name}, " f"found {num_ROIs}."
    )

    # Validation of input/output specs
    wf: napari_workflows.Worfklow = load_workflow(workflow_file)
    if set(wf.leafs()) > set(output_specs.keys()):
        msg = f"Some item of {wf.leafs()=} is not part of {output_specs=}."
        raise ValueError(msg)
    if set(wf.roots()) > set(input_specs.keys()):
        msg = f"Some item of {wf.roots()=} is not part of {input_specs=}."
        raise ValueError(msg)
    list_outputs = sorted(output_specs.keys())

    # Input preparation: "image" type
    image_inputs = [
        (name, params)
        for (name, params) in input_specs.items()
        if params["type"] == "image"
    ]
    input_image_arrays = {}
    if image_inputs:
        img_array = da.from_zarr(f"{in_path}/{component}/{level}")
        # Loop over image inputs and assign corresponding channel of the image
        for (name, params) in image_inputs:
            channel_name = params["channel"]
            if channel_name not in chl_list:
                raise ValueError(f"{channel_name=} not in {chl_list}")
            channel_index = chl_list.index(channel_name)
            input_image_arrays[name] = img_array[channel_index]
            logger.info(f"Prepared input with {name=} and {params=}")
        logger.info(f"{input_image_arrays=}")

    # Input preparation: "label" type
    label_inputs = [
        (name, params)
        for (name, params) in input_specs.items()
        if params["type"] == "label"
    ]
    if label_inputs:
        # Set target_shape for upscaling labels
        if not image_inputs:
            raise ValueError(
                f"{len(label_inputs)=} but num_image_inputs=0, "
                "unclear how to upscale label array(s)."
            )
        target_shape = list(input_image_arrays.values())[0].shape
        # Loop over label inputs and load corresponding (upscaled) image
        input_label_arrays = {}
        for (name, params) in label_inputs:
            label_name = params["label_name"]
            label_array_raw = da.from_zarr(
                f"{in_path}/{component}/labels/{label_name}/{level}"
            )
            input_label_arrays[name] = upscale_array(
                array=label_array_raw, target_shape=target_shape, axis=[1, 2]
            )
            logger.info(f"Prepared input with {name=} and {params=}")
        logger.info(f"{input_label_arrays=}")

    # Output preparation: "label" type
    label_outputs = [
        (name, params)
        for (name, params) in output_specs.items()
        if params["type"] == "label"
    ]
    if label_outputs:
        output_label_zarr_groups = {}
        # Set labels group
        zarrurl = f"{in_path}/{component}"
        if os.path.isdir(f"{zarrurl}/labels"):
            raise NotImplementedError(f"{zarrurl}/labels already exists.")
        labels_group = zarr.group(f"{zarrurl}/labels")
        labels_group.attrs["labels"] = [
            params["label_name"] for (name, params) in label_outputs
        ]

        # Loop over label outputs and (1) set zattrs, (2) create zarr group
        for (name, params) in label_outputs:
            label_name = params["label_name"]

            # (1a) Rescale OME-NGFF datasets (relevant for labeling_level>0)
            new_datasets = rescale_datasets(
                datasets=multiscales[0]["datasets"],
                coarsening_xy=coarsening_xy,
                reference_level=labeling_level,
            )
            # (1b) Write zattrs for specific label
            label_group = labels_group.create_group(label_name)
            label_group.attrs["image-label"] = {
                "version": __OME_NGFF_VERSION__
            }
            label_group.attrs["multiscales"] = [
                {
                    "name": label_name,
                    "version": __OME_NGFF_VERSION__,
                    "axes": [
                        ax
                        for ax in multiscales[0]["axes"]
                        if ax["type"] != "channel"
                    ],
                    "datasets": new_datasets,
                }
            ]
            # (2) Create zarr group for level=0
            store = da.core.get_mapper(
                f"{in_path}/{component}/labels/{label_name}/0"
            )
            mask_zarr = zarr.create(
                shape=img_array[0].shape,
                chunks=img_array[0].chunksize,
                dtype=label_dtype,
                store=store,
                overwrite=False,
                dimension_separator="/",
                # FIXME write_empty_chunks=.. do we need this?
            )
            output_label_zarr_groups[name] = mask_zarr
            logger.info(f"Prepared output with {name=} and {params=}")
        logger.info(f"{output_label_zarr_groups=}")

    # Output preparation: "dataframe" type
    dataframe_outputs = [
        (name, params)
        for (name, params) in output_specs.items()
        if params["type"] == "dataframe"
    ]
    output_dataframe_lists = {}
    for (name, params) in dataframe_outputs:
        output_dataframe_lists[name] = []
        logger.info(f"Prepared output with {name=} and {params=}")
        logger.info(f"{output_dataframe_lists=}")

    #####

    for i_ROI, indices in enumerate(list_indices):
        s_z, e_z, s_y, e_y, s_x, e_x = indices[:]
        region = (slice(s_z, e_z), slice(s_y, e_y), slice(s_x, e_x))
        logger.info(f"ROI {i_ROI+1}/{num_ROIs}: {region=}")

        # Always re-load napari worfklow
        wf: napari_workflows.Worfklow = load_workflow(workflow_file)

        # Set inputs
        for input_name in input_specs.keys():
            input_type = input_specs[input_name]["type"]
            if input_type == "image":
                wf.set(
                    input_name,
                    input_image_arrays[input_name][region].compute(),
                )
            elif input_type == "label":
                wf.set(
                    input_name,
                    input_label_arrays[input_name][region].compute(),
                )

        # Get outputs
        logger.info(f"ROI {i_ROI+1}/{num_ROIs}: wf.set() complete")
        outputs = wf.get(list_outputs)
        logger.info(f"ROI {i_ROI+1}/{num_ROIs}: wf.get() complete")

        # Handle outputs
        for ind_output, output_name in enumerate(list_outputs):
            output_type = output_specs[output_name]["type"]
            if output_type == "dataframe":
                df = outputs[ind_output]
                # Use label column as index, to avoid non-unique indices when
                # using per-FOV labels
                df.index = df["label"].astype(str)
                # Append the new-ROI dataframe to the all-ROIs list
                output_dataframe_lists[output_name].append(df)
            elif output_type == "label":
                mask = outputs[ind_output]
                da.array(mask).to_zarr(
                    url=output_label_zarr_groups[output_name],
                    region=region,
                    compute=True,
                )
        logger.info(f"ROI {i_ROI+1}/{num_ROIs}: output handling complete")

    # Output handling: "dataframe" type (for each output, concatenate ROI
    # dataframes, clean up, and store in a AnnData table on-disk)
    # FIXME: is this cleanup procedure general?
    for (name, params) in dataframe_outputs:
        table_name = params["table_name"]
        list_dfs = output_dataframe_lists[name]
        # Concatenate all FOV dataframes
        df_well = pd.concat(list_dfs, axis=0)
        # Extract labels and drop them from df_well
        labels = pd.DataFrame(df_well["label"].astype(str))
        df_well.drop(labels=["label"], axis=1, inplace=True)
        # Convert all to float (warning: some would be int, in principle)
        measurement_dtype = np.float32
        df_well = df_well.astype(measurement_dtype)
        # Convert to anndata
        measurement_table = ad.AnnData(df_well, dtype=measurement_dtype)
        measurement_table.obs = labels
        # Write to zarr group
        group_tables = zarr.group(f"{in_path}/{component}/tables/")
        write_elem(group_tables, table_name, measurement_table)

    # Output handling: "label" type (for each output, build and write to disk
    # pyramid of coarser levels)
    for (name, params) in label_outputs:
        label_name = params["label_name"]
        build_pyramid(
            zarrurl=f"{zarrurl}/labels/{label_name}",
            overwrite=False,
            num_levels=num_levels,
            coarsening_xy=coarsening_xy,
            chunksize=img_array[0].chunksize,
            aggregation_function=np.max,
        )

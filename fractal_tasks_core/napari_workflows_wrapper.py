"""
Copyright 2022 (C)
    Friedrich Miescher Institute for Biomedical Research and
    University of Zurich

    Original authors:
    Tommaso Comparin <tommaso.comparin@exact-lab.it>
    Marco Franzon <marco.franzon@exact-lab.it>

    This file is part of Fractal and was originally developed by eXact lab
    S.r.l.  <exact-lab.it> under contract with Liberali Lab from the Friedrich
    Miescher Institute for Biomedical Research and Pelkmans Lab from the
    University of Zurich.

Wrapper of napari-workflows
"""
import json
import logging
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Sequence

import anndata as ad
import dask.array as da
import napari_workflows
import numpy as np
import pandas as pd
import zarr
from anndata.experimental import write_elem
from napari_workflows._io_yaml_v1 import load_workflow

import fractal_tasks_core
from fractal_tasks_core.lib_channels import get_channel_from_image_zarr
from fractal_tasks_core.lib_pyramid_creation import build_pyramid
from fractal_tasks_core.lib_regions_of_interest import (
    convert_ROI_table_to_indices,
)
from fractal_tasks_core.lib_upscale_array import upscale_array
from fractal_tasks_core.lib_zattrs_utils import extract_zyx_pixel_sizes
from fractal_tasks_core.lib_zattrs_utils import rescale_datasets


__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__


logger = logging.getLogger(__name__)


class OutOfTaskScopeError(NotImplementedError):
    """
    Encapsulates features that are out-of-scope for the current wrapper task
    """

    pass


def napari_workflows_wrapper(
    *,
    # Default arguments for fractal tasks:
    input_paths: Sequence[Path],
    output_path: Path,
    component: str,
    metadata: Dict[str, Any],
    # Task-specific arguments:
    workflow_file: str,
    input_specs: Dict[str, Dict[str, str]],
    output_specs: Dict[str, Dict[str, str]],
    ROI_table_name: str = "FOV_ROI_table",
    level: int = 0,
    relabeling: bool = True,
    expected_dimensions: int = 3,
):
    """
    Run a napari-workflow on the ROIs of a single OME-NGFF image

    Full documentation for all arguments is still TBD, especially because some
    of them are standard arguments for Fractal tasks that should be documented
    in a standard way. Here are some examples::

        input_paths = ["/some/path/*.zarr"]
        output_path = "/some/path/*.zarr"
        component = "some_plate.zarr/B/03/0"
        metadata = {"num_levels": 4, "coarsening_xy": 2}

        # Examples of allowed entries for input_specs and output_specs
        input_specs = {
            "in_1": {"type": "image", "wavelength_id": "A01_C02"},
            "in_2": {"type": "image", "channel_label": "DAPI"},
            "in_3": {"type": "label", "label_name": "label_DAPI"},
        }
        output_specs = {
            "out_1": {"type": "label", "label_name": "label_DAPI_new"},
            "out_2": {"type": "dataframe", "table_name": "measurements"},
        }

    :param input_paths: TBD (default arg for Fractal tasks)
    :param output_path: TBD (default arg for Fractal tasks)
    :param metadata: TBD (default arg for Fractal tasks)
    :param component: TBD (default arg for Fractal tasks)
    :param workflow_file: Absolute path to napari-workflows YAML file
    :param input_specs: See examples above.
    :param output_specs: See examples above.

    :param level: Pyramid level of the image to be segmented.
    :param expected_dimensions: Expected dimensions (either 2 or 3).

    :param relabeling: If ``True``, apply relabeling so that label values are
                       unique across ROIs.
    :param ROI_table_name: name of the table that contains ROIs to which the\
                          task applies the napari-worfklow
    """

    wf: napari_workflows.Worfklow = load_workflow(workflow_file)
    logger.info(f"Loaded workflow from {workflow_file}")

    # Validation of input/output specs
    if not (set(wf.leafs()) <= set(output_specs.keys())):
        msg = f"Some item of {wf.leafs()=} is not part of {output_specs=}."
        logger.warning(msg)
    if not (set(wf.roots()) <= set(input_specs.keys())):
        msg = f"Some item of {wf.roots()=} is not part of {input_specs=}."
        logger.error(msg)
        raise ValueError(msg)
    list_outputs = sorted(output_specs.keys())

    # Characterization of workflow and scope restriction
    input_types = [params["type"] for (name, params) in input_specs.items()]
    output_types = [params["type"] for (name, params) in output_specs.items()]
    are_inputs_all_images = set(input_types) == {"image"}
    are_outputs_all_labels = set(output_types) == {"label"}
    are_outputs_all_dataframes = set(output_types) == {"dataframe"}
    is_labeling_workflow = are_inputs_all_images and are_outputs_all_labels
    is_measurement_only_workflow = are_outputs_all_dataframes
    # Level-related constraint
    logger.info(f"This workflow acts at {level=}")
    logger.info(
        f"Is the current workflow a labeling one? {is_labeling_workflow}"
    )
    if level > 0 and not is_labeling_workflow:
        msg = (
            f"{level=}>0 is currently only accepted for labeling workflows, "
            "i.e. those going from image(s) to label(s)"
        )
        logger.error(msg)
        raise OutOfTaskScopeError(msg)
    # Relabeling-related (soft) constraint
    if is_measurement_only_workflow and relabeling:
        logger.warning(
            "This is a measurement-output-only workflow, setting "
            "relabeling=False."
        )
        relabeling = False
    if relabeling:
        max_label_for_relabeling = 0

    # Pre-processing of task inputs
    if len(input_paths) > 1:
        raise NotImplementedError("We currently only support a single in_path")
    in_path = input_paths[0].parent.as_posix()
    num_levels = metadata["num_levels"]
    coarsening_xy = metadata["coarsening_xy"]
    label_dtype = np.uint32

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
        level=level,
        coarsening_xy=coarsening_xy,
        full_res_pxl_sizes_zyx=full_res_pxl_sizes_zyx,
    )
    num_ROIs = len(list_indices)
    logger.info(
        f"Completed reading ROI table {ROI_table_name}, " f"found {num_ROIs}."
    )

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
            if "wavelength_id" in params and "channel_label" in params:
                raise ValueError(
                    "One and only one among channel_label and wavelength_id"
                    f" attributes must be provided, but input {name} in "
                    f"input_specs has {params=}."
                )
            channel = get_channel_from_image_zarr(
                image_zarr_path=f"{in_path}/{component}",
                wavelength_id=params.get("wavelength_id", None),
                label=params.get("channel_label", None),
            )
            channel_index = channel["index"]
            input_image_arrays[name] = img_array[channel_index]

            # Handle dimensions
            shape = input_image_arrays[name].shape
            if expected_dimensions == 3 and shape[0] == 1:
                logger.warning(
                    f"Input {name} has shape {shape} "
                    f"but {expected_dimensions=}"
                )
            if expected_dimensions == 2:
                if shape[0] == 1:
                    input_image_arrays[name] = input_image_arrays[name][
                        0, :, :
                    ]
                else:
                    msg = (
                        f"Input {name} has shape {shape} "
                        f"but {expected_dimensions=}"
                    )
                    logger.error(msg)
                    raise ValueError(msg)
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
            logger.warning(
                f"{len(label_inputs)=} but num_image_inputs=0. "
                "Label array(s) will not be upscaled."
            )
            upscale_labels = False
        else:
            target_shape = list(input_image_arrays.values())[0].shape
            upscale_labels = True
        # Loop over label inputs and load corresponding (upscaled) image
        input_label_arrays = {}
        for (name, params) in label_inputs:
            label_name = params["label_name"]
            label_array_raw = da.from_zarr(
                f"{in_path}/{component}/labels/{label_name}/{level}"
            )
            input_label_arrays[name] = label_array_raw
            if upscale_labels:
                input_label_arrays[name] = upscale_array(
                    array=input_label_arrays[name],
                    target_shape=target_shape,
                    axis=[1, 2],
                    pad_with_zeros=True,
                )
            # Handle dimensions
            shape = input_label_arrays[name].shape
            if expected_dimensions == 3 and shape[0] == 1:
                logger.warning(
                    f"Input {name} has shape {shape} "
                    f"but {expected_dimensions=}"
                )
            if expected_dimensions == 2:
                if shape[0] == 1:
                    input_label_arrays[name] = input_label_arrays[name][
                        0, :, :
                    ]
                else:
                    msg = (
                        f"Input {name} has shape {shape} "
                        f"but {expected_dimensions=}"
                    )
                    logger.error(msg)
                    raise ValueError(msg)
            logger.info(f"Prepared input with {name=} and {params=}")
        logger.info(f"{input_label_arrays=}")

    # Output preparation: "label" type
    label_outputs = [
        (name, params)
        for (name, params) in output_specs.items()
        if params["type"] == "label"
    ]
    if label_outputs:
        # Preliminary scope checks
        if len(label_outputs) > 1:
            raise OutOfTaskScopeError(
                "Multiple label outputs would break label-inputs-only "
                f"workflows (found {len(label_outputs)=})."
            )
        if len(label_outputs) > 1 and relabeling:
            raise OutOfTaskScopeError(
                "Multiple label outputs would break relabeling in labeling+"
                f"measurement workflows (found {len(label_outputs)=})."
            )

        # We only support two cases:
        # 1. If there exist some input images, then use the first one to
        #    determine output-label array properties
        # 2. If there are no input images, but there are input labels, then (A)
        #    re-load the pixel sizes and re-build ROI indices, and (B) use the
        #    first input label to determine output-label array properties
        if image_inputs:
            reference_array = list(input_image_arrays.values())[0]
        elif label_inputs:
            reference_array = list(input_label_arrays.values())[0]
            # Re-load pixel size, matching to the correct level
            input_label_name = label_inputs[0][1]["label_name"]
            zattrs_file = (
                f"{in_path}/{component}/labels/{input_label_name}/.zattrs"
            )
            # Read pixel sizes from zattrs file
            full_res_pxl_sizes_zyx = extract_zyx_pixel_sizes(
                zattrs_file, level=0
            )
            # Create list of indices for 3D FOVs spanning the whole Z direction
            list_indices = convert_ROI_table_to_indices(
                ROI_table,
                level=level,
                coarsening_xy=coarsening_xy,
                full_res_pxl_sizes_zyx=full_res_pxl_sizes_zyx,
            )
            num_ROIs = len(list_indices)
            logger.info(
                f"Re-loaded reading ROI table {ROI_table_name}, "
                f"and created indices using {full_res_pxl_sizes_zyx=}. "
                "This is necessary because label-input-only workflows may "
                "have label inputs that are at a different resolution and "
                "are not upscaled."
            )
        else:
            msg = (
                "Missing image_inputs and label_inputs, we cannot assign"
                " label output properties"
            )
            raise OutOfTaskScopeError(msg)

        # Extract label properties from reference_array, and make sure they are
        # for three dimensions
        label_shape = reference_array.shape
        label_chunksize = reference_array.chunksize
        if len(label_shape) == 2 and len(label_chunksize) == 2:
            if expected_dimensions == 3:
                raise ValueError(
                    f"Something wrong: {label_shape=} but "
                    f"{expected_dimensions=}"
                )
            label_shape = (1, label_shape[0], label_shape[1])
            label_chunksize = (1, label_chunksize[0], label_chunksize[1])
        logger.info(f"{label_shape=}")
        logger.info(f"{label_chunksize=}")

        # Create labels zarr group and combine existing/new labels in .zattrs
        new_labels = [params["label_name"] for (name, params) in label_outputs]
        zarrurl = f"{in_path}/{component}"
        try:
            with open(f"{zarrurl}/labels/.zattrs", "r") as f_zattrs:
                existing_labels = json.load(f_zattrs)["labels"]
        except FileNotFoundError:
            existing_labels = []
        intersection = set(new_labels) & set(existing_labels)
        logger.info(f"{new_labels=}")
        logger.info(f"{existing_labels=}")
        if intersection:
            raise OutOfTaskScopeError(
                f"Labels {intersection} already exist "
                "but are part of outputs"
            )
        labels_group = zarr.group(f"{zarrurl}/labels")
        labels_group.attrs["labels"] = existing_labels + new_labels

        # Loop over label outputs and (1) set zattrs, (2) create zarr group
        output_label_zarr_groups: Dict[str, Any] = {}
        for (name, params) in label_outputs:
            label_name = params["label_name"]

            # (1a) Rescale OME-NGFF datasets (relevant for level>0)
            new_datasets = rescale_datasets(
                datasets=multiscales[0]["datasets"],
                coarsening_xy=coarsening_xy,
                reference_level=level,
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
            # (2) Create zarr group at level=0
            store = da.core.get_mapper(
                f"{in_path}/{component}/labels/{label_name}/0"
            )
            mask_zarr = zarr.create(
                shape=label_shape,
                chunks=label_chunksize,
                dtype=label_dtype,
                store=store,
                overwrite=False,
                dimension_separator="/",
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
    output_dataframe_lists: Dict[str, List] = {}
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
        wf = load_workflow(workflow_file)

        # Set inputs
        for input_name in input_specs.keys():
            input_type = input_specs[input_name]["type"]
            # Handle expected_dimensions
            if expected_dimensions == 2:
                actual_region = region[1:]
            else:
                actual_region = region

            if input_type == "image":
                wf.set(
                    input_name,
                    input_image_arrays[input_name][actual_region].compute(),
                )
            elif input_type == "label":
                wf.set(
                    input_name,
                    input_label_arrays[input_name][actual_region],
                )

        # Get outputs
        outputs = wf.get(list_outputs)

        # Iterate first over dataframe outputs (to use the correct
        # max_label_for_relabeling, if needed)
        for ind_output, output_name in enumerate(list_outputs):
            if output_specs[output_name]["type"] != "dataframe":
                continue
            df = outputs[ind_output]
            if relabeling:
                df["label"] += max_label_for_relabeling
                logger.info(
                    f'ROI {i_ROI+1}/{num_ROIs}: Relabeling "{name}" dataframe'
                    "output, with {max_label_for_relabeling=}"
                )

            # Append the new-ROI dataframe to the all-ROIs list
            output_dataframe_lists[output_name].append(df)

        # After all dataframe outputs, iterate over label outputs (which
        # actually can be only 0 or 1)
        for ind_output, output_name in enumerate(list_outputs):
            if output_specs[output_name]["type"] != "label":
                continue
            mask = outputs[ind_output]

            # Check dimensions
            if len(mask.shape) != expected_dimensions:
                msg = (
                    f"Output {output_name} has shape {mask.shape} "
                    f"but {expected_dimensions=}"
                )
                logger.error(msg)
                raise ValueError(msg)
            elif expected_dimensions == 2:
                mask = np.expand_dims(mask, axis=0)

            # Sanity check: issue warning for non-consecutive labels
            unique_labels = np.unique(mask)
            num_unique_labels_in_this_ROI = len(unique_labels)
            if np.min(unique_labels) == 0:
                num_unique_labels_in_this_ROI -= 1
            num_labels_in_this_ROI = int(np.max(mask))
            if num_labels_in_this_ROI != num_unique_labels_in_this_ROI:
                logger.warning(
                    f'ROI {i_ROI+1}/{num_ROIs}: "{name}" label output has'
                    f"non-consecutive labels: {num_labels_in_this_ROI=} but"
                    f"{num_unique_labels_in_this_ROI=}"
                )

            if relabeling:
                mask[mask > 0] += max_label_for_relabeling
                logger.info(
                    f'ROI {i_ROI+1}/{num_ROIs}: Relabeling "{name}" label '
                    f"output, with {max_label_for_relabeling=}"
                )
                max_label_for_relabeling += num_labels_in_this_ROI
                logger.info(
                    f"ROI {i_ROI+1}/{num_ROIs}: label-number update with "
                    f"{num_labels_in_this_ROI=}; "
                    f"new {max_label_for_relabeling=}"
                )

            da.array(mask).to_zarr(
                url=output_label_zarr_groups[output_name],
                region=region,
                compute=True,
            )
        logger.info(f"ROI {i_ROI+1}/{num_ROIs}: output handling complete")

    # Output handling: "dataframe" type (for each output, concatenate ROI
    # dataframes, clean up, and store in a AnnData table on-disk)
    for (name, params) in dataframe_outputs:
        table_name = params["table_name"]
        # Concatenate all FOV dataframes
        list_dfs = output_dataframe_lists[name]
        df_well = pd.concat(list_dfs, axis=0, ignore_index=True)
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
            chunksize=label_chunksize,
            aggregation_function=np.max,
        )


if __name__ == "__main__":
    from pydantic import BaseModel
    from fractal_tasks_core._utils import run_fractal_task

    class TaskArguments(BaseModel):
        input_paths: Sequence[Path]
        output_path: Path
        metadata: Dict[str, Any]
        component: str
        workflow_file: str
        input_specs: Dict[str, Dict[str, str]]
        output_specs: Dict[str, Dict[str, str]]
        ROI_table_name: str = "FOV_ROI_table"
        level: int = 0
        relabeling: bool = True
        expected_dimensions: int = 3

    run_fractal_task(
        task_function=napari_workflows_wrapper,
        TaskArgsModel=TaskArguments,
        logger_name=logger.name,
    )

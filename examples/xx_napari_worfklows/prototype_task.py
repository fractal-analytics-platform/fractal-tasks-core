import anndata as ad
import dask.array as da
import napari_workflows
import numpy as np
import pandas as pd
import zarr
from anndata.experimental import write_elem
from napari_workflows._io_yaml_v1 import load_workflow

from ..lib_pyramid_creation import build_pyramid


def napari_workflows_wrapper():

    in_path = "tmp_out"
    component = "20200812-CardiomyocyteDifferentiation14-Cycle1.zarr/B/03/0"

    level = 0
    num_levels = 6
    coarsening_xy = 2

    channel_mapping = {"DAPI": 0}

    """
    workflow_file = "wf_7.yaml"
    input_specs = {
        "dapi_img": dict(in_type="image", channel_name="DAPI"),
        "dapi_label_img": dict(in_type="label", table_name="label_DAPI"),
        "lamin_img": dict(in_type="image", channel_name="DAPI"),
        "lamin_label_img": dict(in_type="label", table_name="label_DAPI"),
    }
    output_specs = {
        "regionprops_DAPI": dict(
            out_type="dataframe", table_name="dapi_measurements"
        ),
        "regionprops_Lamin": dict(
            out_type="dataframe", table_name="lamin_measurements"
        ),
    }
    """

    workflow_file = "wf_6.yaml"
    input_specs = {
        "slice_img": dict(in_type="image", channel_name="DAPI"),
        "slice_img_c2": dict(in_type="image", channel_name="DAPI"),
    }
    output_specs = {
        "Result of Expand labels (scikit-image, nsbatwm)": dict(
            out_type="label",
            table_name="label_DAPI_nw",
        ),
        "regionprops_DAPI": dict(
            out_type="dataframe", table_name="dapi_measurements"
        ),
    }

    list_outputs = sorted(output_specs.keys())

    # FIXME: use standard ROIs
    list_indices = [
        [0, 10, 0, 2160, 0, 2560],
        [0, 10, 0, 2160, 2560, 5120],
    ]

    # Validation of specs
    wf: napari_workflows.Worfklow = load_workflow(workflow_file)
    assert set(wf.leafs()) == set(output_specs)
    assert set(wf.roots()) == set(input_specs)

    # This loads /some/path/plate.zarr/B/03/0/{level}
    img_array = da.from_zarr(f"{in_path}/{component}/{level}")

    # Prepare label arrays for input
    required_labels = [
        input_specs[input_item]["table_name"]
        for input_item in input_specs
        if input_specs[input_item]["in_type"] == "label"
    ]
    label_arrays = {}
    for label_name in required_labels:
        label_array_raw = da.from_zarr(
            f"{in_path}/{component}/labels/{label_name}/{level}"
        )
        upscale_factor = img_array.shape[-1] // label_array_raw.shape[-1]
        assert (
            upscale_factor == img_array.shape[-2] // label_array_raw.shape[-2]
        )
        label_img_up_x = np.repeat(label_array_raw, upscale_factor, axis=2)
        label_arrays[label_name] = np.repeat(
            label_img_up_x, upscale_factor, axis=1
        )

    # Prepare label arrays for output
    required_labels = [
        output_specs[output_item]["table_name"]
        for output_item in output_specs
        if output_specs[output_item]["out_type"] == "label"
    ]
    # Open new zarr group for each mask 0-th level
    label_zarr_groups = {}
    for label_name in required_labels:

        label_dtype = np.uint32
        zarrurl = f"{in_path}/{component}"

        XXXX

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
                    ax
                    for ax in multiscales[0]["axes"]
                    if ax["type"] != "channel"
                ],
                "datasets": new_datasets,
            }
        ]

        XXX

        zarr.group(f"{zarrurl}/labels")
        zarr.group(f"{zarrurl}/labels/{label_name}")
        store = da.core.get_mapper(f"{zarrurl}/labels/{label_name}/0")
        label_dtype = np.uint32
        mask_zarr = zarr.create(
            shape=img_array[0].shape,
            chunks=img_array[0].chunksize,
            dtype=label_dtype,
            store=store,
            overwrite=False,
            dimension_separator="/",
            # FIXME write_empty_chunks=.. do we need this?
        )
        label_zarr_groups[label_name] = mask_zarr

    # Prepare dataframe-output collections
    required_dataframes = [
        output_specs[output_item]["table_name"]
        for output_item in output_specs
        if output_specs[output_item]["out_type"] == "dataframe"
    ]
    tables = {}
    for table_name in required_dataframes:
        tables[table_name] = []

    for indices in list_indices:
        s_z, e_z, s_y, e_y, s_x, e_x = indices[:]
        region = (slice(s_z, e_z), slice(s_y, e_y), slice(s_x, e_x))
        print(region)

        # FIXME do I have to re-load it each time?
        wf: napari_workflows.Worfklow = load_workflow(workflow_file)

        # Handle inputs
        for input_name in input_specs.keys():
            input_type = input_specs[input_name]["in_type"]

            if input_type == "image":
                channel = input_specs[input_name]["channel_name"]
                wf.set(input_name, img_array[channel_mapping[channel]][region])

            elif input_type == "label":
                label_name = input_specs[input_name]["table_name"]
                assert label_name in label_arrays.keys()
                wf.set(input_name, label_arrays[label_name][region])

        # Get outputs
        outputs = wf.get(list_outputs)

        # Handle outputs
        for ind_output, output_name in enumerate(list_outputs):
            output_type = output_specs[output_name]["out_type"]
            if output_type == "dataframe":
                table_name = output_specs[output_name]["table_name"]
                # Select output dataframe
                df = outputs[ind_output]
                # Use label column as index, simply to avoid non-unique indices when
                # using per-FOV labels
                df.index = df["label"].astype(str)
                # FIXME: add relabeling
                # Append the new-ROI dataframe to the all-ROIs list
                tables[table_name].append(df)
            elif output_type == "label":
                table_name = output_specs[output_name]["table_name"]
                mask = outputs[ind_output]
                # FIXME: add relabeling
                # if relabeling:
                #    mask += tot_labels[table_name]
                #    tot_labels[table_name] += np.max(mask)
                da.array(mask).to_zarr(
                    url=label_zarr_groups[label_name],
                    region=region,
                    compute=True,
                )

    # For each dataframe output: concatenate all ROI dataframes, clean up, and
    # store in a AnnData table
    # FIXME: is this cleanup procedure general?
    for table_name in required_dataframes:
        list_dfs = tables[table_name]
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

    # For each output label, build and write to disk a pyramid of coarser levels
    for label_name in required_labels:
        build_pyramid(
            zarrurl=f"{zarrurl}/labels/{label_name}",
            overwrite=False,
            num_levels=num_levels,
            coarsening_xy=coarsening_xy,
            chunksize=img_array[0].chunksize,
            aggregation_function=np.max,
        )


if __name__ == "__main__":
    napari_workflows_wrapper()

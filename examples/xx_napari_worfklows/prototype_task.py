import dask.array as da
import napari_workflows
import numpy as np
from napari_workflows._io_yaml_v1 import load_workflow


def napari_workflows_wrapper():

    in_path = "tmp_out"
    component = "20200812-CardiomyocyteDifferentiation14-Cycle1.zarr/B/03/0"
    level = 0

    channel_mapping = {"DAPI": 0}

    workflow_file = "wf.yaml"

    input_specs = {
        "dapi_img": dict(in_type="image", channel_name="DAPI"),
        "dapi_label_img": dict(in_type="label", table_name="label_DAPI"),
    }
    output_specs = {
        "DAPI_regionprops": dict(
            out_type="dataframe", table_name="dapi_measurements"
        ),
    }
    list_outputs = sorted(output_specs.keys())

    # FIXME: use standard ROIs
    list_indices = [
        [0, 10, 0, 2160, 0, 2560],
    ]

    # Validation of specs
    wf: napari_workflows.Worfklow = load_workflow(workflow_file)
    assert len(wf.leafs()) == len(output_specs)
    assert len(wf.roots()) == len(input_specs)

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

    # FIXME: prepare label arrays for output
    required_labels = [
        output_specs[output_item]["table_name"]
        for output_item in output_specs
        if output_specs[output_item]["out_type"] == "label"
    ]

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
                tables[table_name].append(outputs[ind_output])
            # FIXME: handle label output

    # FIXME write each (concatenated) dataframes to a table


if __name__ == "__main__":
    napari_workflows_wrapper()

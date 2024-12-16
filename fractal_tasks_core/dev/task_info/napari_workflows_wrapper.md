### Purpose
- Executes a **napari workflow** on the regions of interest (ROIs) within a single OME-NGFF image.
- Processes specified images and labels as inputs to the workflow, producing outputs such as new labels and data tables.
- Offers **flexibility in defining input and output** specifications to customize the workflow for specific datasets and analysis needs.

### Limitations
- Currently supports only Napari workflows that utilize functions from the `napari-segment-blobs-and-things-with-membranes` module. Other Napari-compatible modules are not supported.

### Input Specifications
Napari workflows require explicit definitions of input and output data.
Example of valid `input_specs`:
```json
{
    "in_1": {"type": "image", "channel": {"wavelength_id": "A01_C02"}},
    "in_2": {"type": "image", "channel": {"label": "DAPI"}},
    "in_3": {"type": "label", "label_name": "label_DAPI"}
}
```

Example of valid `output_specs`:
```json
{
    "out_1": {"type": "label", "label_name": "label_DAPI_new"},
    "out_2": {"type": "dataframe", "table_name": "measurements"},
}
```

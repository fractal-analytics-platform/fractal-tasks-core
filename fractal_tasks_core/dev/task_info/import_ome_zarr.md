### Purpose
- Imports a **single OME-Zarr dataset** into the Fractal framework for further processing.
- Supports importing either a **full OME-Zarr HCS plate** or an **individual OME-Zarr image**.
- Ensures the OME-Zarr dataset is located in the `zarr_dir` specified by the dataset.
- Generates the necessary **image list metadata** required for processing the OME-Zarr with Fractal.
- Optionally **adds new ROI tables** to the existing OME-Zarr, enabling compatibility with many other tasks.

### Limitations
- Supports only OME-Zarr datasets already present in the `zarr_dir` of the corresponding dataset.
- Assumes the input OME-Zarr is correctly structured and formatted for compatibility with the Fractal framework.

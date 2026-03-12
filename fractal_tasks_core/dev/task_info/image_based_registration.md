### Purpose
- **Computes image-based registration** transformations for acquisitions in **HCS** OME-Zarr datasets.
- Processes images grouped by well, under the assumption that each well contains one image per acquisition.
- Calculates transformations for **specified regions of interest (ROIs)** and stores the results in the corresponding ROI table.
- Typically used as the first task in a workflow, followed by `Find Registration Consensus` and optionally `Apply Registration to Image`.

### Limitations
- Supports only HCS OME-Zarr datasets, leveraging their acquisition metadata and well-based image grouping.
- Assumes each well contains a single image per acquisition.

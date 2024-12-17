### Purpose
- Determines the **consensus alignment** region across all selected acquisitions **within each well of an HCS OME-Zarr dataset**.
- Generates a new ROI table for each image, defining consensus regions that are aligned across all acquisitions.
- Typically used as the second task in a workflow, following `Calculate Registration (image-based)` and optionally preceding `Apply Registration to Image`.

### Limitations
- Supports only HCS OME-Zarr datasets, leveraging their acquisition metadata and well-based image grouping.

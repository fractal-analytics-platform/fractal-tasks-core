### Purpose
- Converts **2D and 3D images from CellVoyager CV7000/8000** systems into OME-Zarr format, creating OME-Zarr HCS plates and combining all fields of view in a well into a single image.
- Saves Fractal **region-of-interest (ROI) tables** for both individual fields of view and the entire well.
- Handles overlapping fields of view by adjusting their positions to be non-overlapping while retaining the original position data as additional columns in the ROI tables.
- Supports processing multiple plates in a single task.

### Limitations
- Currently, this task does not support time-resolved data and ignores the time fields in CellVoyager metadata.

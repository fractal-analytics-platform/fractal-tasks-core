### Purpose
- Converts **multiplexed 2D and 3D images from CellVoyager CV7000/8000** systems into OME-Zarr format, storing each acquisition as a separate OME-Zarr image in the same OME-Zarr plate.
- Creates **OME-Zarr HCS plates**, combining all fields of view for each acquisition in a well into a single image.
- Saves Fractal **region-of-interest (ROI) tables** for both individual fields of view and the entire well.
- Handles overlapping fields of view by adjusting their positions to be non-overlapping, while preserving the original position data as additional columns in the ROI tables.

### Limitations
- This task currently does not support time-resolved data and ignores the time fields in CellVoyager metadata.

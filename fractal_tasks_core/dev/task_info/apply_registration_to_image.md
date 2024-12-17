### Purpose
- **Applies pre-calculated registration** transformations to images in an **HCS** OME-Zarr dataset, aligning all acquisitions to a specified reference acquisition.
- **Masks regions not included** in the registered ROI table and aligns both intensity and label images.
- Replaces the non-aligned image with the newly aligned image in the dataset if `overwrite input` is selected.
- Typically used as the third task in a workflow, following `Calculate Registration (image-based)` and `Find Registration Consensus`.

### Limitations
- If `overwrite input` is selected, the non-aligned image is permanently deleted, which may impact workflows requiring access to the original images.

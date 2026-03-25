### Purpose
- Extracts region-properties features (shape and intensity) from a label image
  using `skimage.measure.regionprops_table` and saves the result as a
  `FeatureTable` inside the same OME-Zarr container.
- Supports both **2D** (YX) and **3D** (ZYX) images.
- Two built-in feature groups are available: `ShapeFeatures` and
  `IntensityFeatures`.

### Limitations
- Requires a label image to already exist in the OME-Zarr container.
- Processes one label image at a time.

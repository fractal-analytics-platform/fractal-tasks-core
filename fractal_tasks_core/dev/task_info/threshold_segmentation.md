### Purpose
- Segments intensity images by applying a threshold and labelling connected components.
- Supports **automatic (Otsu)** and **manual** threshold selection.
- Optionally applies pre-processing filters (Gaussian, Median) and post-processing
  (size filtering) around the threshold step.
- Writes the result as a label image inside the same OME-Zarr container.

### Limitations
- Operates on a single input channel at a time.
- Requires images to have a channel axis (CZYX or CYX).

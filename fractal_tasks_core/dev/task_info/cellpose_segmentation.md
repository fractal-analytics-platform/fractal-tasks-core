### Purpose
- **Segments images using Cellpose models**.
- Supports both **built-in Cellpose models** (shipped with Cellpose) and **user-trained models**.
- Accepts dual image input for segmentation.
- Can process **arbitrary regions of interest (ROIs)**, including whole images, fields of view (FOVs), or masked outputs from prior segmentations, based on corresponding ROI tables.
- Provides access to all advanced Cellpose parameters.
- Allows custom rescaling options per channel, particularly useful for sparse images.

### Limitations
- Compatible only with Cellpose 2.x models; does not yet support 3.x models.

### Purpose
- Performs **projection of intensity images** with respect to **Z**, **Y**, or **X** using a specified projection method.
- **Generates a new OME-Zarr HCS plate** to store the projected data.
- Supported projection methods: Maximum Intensity Projection (MIP), Minimum Intensity Projection (MinIP), Mean Intensity Projection (MeanIP), and Sum Intensity Projection (SumIP).
- Allows for optional Z-axis upscaling when doing **Y** or **X** projections to compensate for a typical lower resolution in Z dimension.
- Allows for optional autofocus based on image sharpness that makes projection only within a defined radius around the sharpest plane.

### Limitations
- Supports projections only for OME-Zarr HCS plates; other collections of OME-Zarrs are not yet supported.

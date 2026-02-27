### Purpose
- Performs **Z-axis projection of intensity images** using a specified projection method.
- Stores the result in a **new OME-Zarr file** next to the input.

### Output location
The output zarr is placed alongside the input with the projection method appended to its
name. For example, an input of `my_image.zarr` with method `mip` produces `my_image_mip.zarr`.

### Limitations
- The input zarr URL must end with `.zarr`.
- Supports projections for any 3D OME-Zarr image (ZYX, CZYX, TCZYX, etc.).

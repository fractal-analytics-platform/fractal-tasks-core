### Purpose
- **Corrects illumination** in OME-Zarr images using **pre-calculated flatfield profiles**.
- Optionally performs **background subtraction** using **pre-calculated darkfield profiles** or constant values.
- Corrections are applied with a formula:
  - `corrected_image = (raw_image - background) / illumination`

### Limitations
- Requires pre-calculated flatfield profiles in TIFF format.
- Optional background subtraction requires pre-calculated darkfield profiles in TIFF format or constant values.
- Flatfield profiles are always normalized to 0-1 range before correction.
- Requires all channels to have corresponding illumination profiles. If background correction is used, all channels must also have corresponding background profiles or constant values.

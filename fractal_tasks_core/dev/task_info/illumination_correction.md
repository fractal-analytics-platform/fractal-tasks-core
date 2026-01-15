### Purpose
- **Corrects illumination** in OME-Zarr images using **pre-calculated flatfield profiles**.
- Optionally performs **background profiles subtraction** using **pre-calculated darkfield profiles**.
- Optionally performs **background subtraction** using a constant value.
- Corrections are applied with a formula:
  - `corrected_image = (raw_image - background) / illumination`

### Limitations
- Requires pre-calculated flatfield profiles in TIFF format.
- Optional background profile subtraction requires pre-calculated darkfield profiles in TIFF format.
- Flatfield profiles are always normalized to 0-1 range before correction.
- Requires all channels to have corresponding illumination profiles.

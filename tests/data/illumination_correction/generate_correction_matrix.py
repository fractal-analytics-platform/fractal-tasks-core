import numpy as np
from PIL import Image

# run this script in it's own folder with python generate_correction_matrix.py
# we want simple to understand rather than very realistic corrections

image_shape = (2160, 2560)

# old test data used in tasks(v1) and tests(v2)
illum_corr = np.ones(image_shape, dtype=np.uint16)
Image.fromarray(illum_corr).save("illum_corr_matrix.png", optimize=True)

# illumination correction test data used in tasks(v2)
# modeled as gaussian falloff, 
# because any constant illumination correction does nothing to the image
X, Y = np.meshgrid(
    np.linspace(-1, 1, image_shape[1]),
    np.linspace(-1, 1, image_shape[0]),
)
radius = np.sqrt(X**2 + Y**2)
flatfield = np.exp(-(radius**2)) * 20000  
flatfield = flatfield.astype(np.uint16)
Image.fromarray(flatfield).save("flatfield_corr_matrix.png", optimize=True)

# background correction test data used in tasks(v2)
darkfield = 7 * np.ones(image_shape, dtype=np.uint16)
Image.fromarray(darkfield).save("darkfield_corr_matrix.png", optimize=True)
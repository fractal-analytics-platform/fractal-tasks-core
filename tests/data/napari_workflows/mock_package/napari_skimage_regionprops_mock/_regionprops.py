import warnings

import numpy as np
import pandas

try:
    import napari
except Exception as e:
    warnings.warn(str(e))

from napari_tools_menu import register_function


@register_function(menu="Measurement tables > Regionprops (scikit-image, nsr)")
def regionprops_table(
    image: "napari.types.ImageData",
    labels: "napari.types.LabelsData",
    size: bool = True,
    intensity: bool = True,
    perimeter: bool = False,
    shape: bool = False,
    position: bool = False,
    moments: bool = False,
    napari_viewer: "napari.Viewer" = None,
) -> "pandas.DataFrame":
    """
    MOCK OF THE ORIGINAL FUNCTION
    """

    num_labels = len(np.unique(labels))
    int_values = np.arange(num_labels)
    float_values = np.arange(num_labels) + 0.5
    table = dict(
        label=int_values,
        area=float_values,
        bbox_area=float_values,
        equivalent_diameter=float_values,
        convex_area=float_values,
    )
    df = pandas.DataFrame(table)
    return df

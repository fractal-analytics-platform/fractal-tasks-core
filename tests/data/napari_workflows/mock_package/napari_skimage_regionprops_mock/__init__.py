import logging

import numpy as np
import pandas

logging.critical(
    "Using a mocked version of "
    "napari_skimage_regionprops, which only exposes a "
    "mock-up of the `regionprops_table` function."
)


def regionprops_table(
    image,
    labels,
    size: bool = True,
    intensity: bool = True,
    perimeter: bool = False,
    shape: bool = False,
    position: bool = False,
    moments: bool = False,
    napari_viewer=None,
) -> pandas.DataFrame:
    """
    This is a mock of the original regionprops_table from
    napari_skimage_regionprops, only used for testing.
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

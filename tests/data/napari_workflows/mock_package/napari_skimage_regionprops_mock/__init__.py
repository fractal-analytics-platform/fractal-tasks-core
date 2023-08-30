import logging

from napari_plugin_engine import napari_hook_implementation

from ._regionprops import regionprops_table  # , regionprops_table_all_frames

logging.critical(
    "WARNING: THIS IS A MOCK OF THE ACTUAL 'napari_skimage_regionprops' PACKAGE"
)


@napari_hook_implementation
def napari_experimental_provide_function():
    return [regionprops_table, visualize_measurement_on_labels, load_csv]

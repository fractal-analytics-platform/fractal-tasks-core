import importlib

import pytest

modules = [
    "napari_workflows",
    "napari_skimage_regionprops",
    "napari_skimage_regionprops._regionprops",
    "napari_segment_blobs_and_things_with_membranes",
]


@pytest.mark.parametrize("module", modules)
def test_imports(module):
    importlib.import_module(module)

This is a mock of the original
[`napari_skimage_regionprops`](https://github.com/haesleinhuepf/napari-skimage-regionprops)
package. The mock keeps the original package structure, but it only exposes a
single function (`regionprops_table`), which is emptied by its actual original
contents.

The original package is released under BSD-3 licence with copyright notice:
Copyright (c) 2021, Robert Haase, Physics of Life, TU Dresden
(see https://github.com/haesleinhuepf/napari-skimage-regionprops/blob/master/LICENSE)


To re-build the wheel file for this package, use
```
poetry run python -m build --wheel
```

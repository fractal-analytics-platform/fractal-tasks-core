To inspect one of the YAML workflow files, you can use the `inspect_workflow.py` script. Example:
```
$ python inspect_workflow.py  wf_2.yaml

[... warnings about QT ...]

Workflow:
Result of Expand labels (scikit-image, nsbatwm) <- (<function expand_labels at 0x7f7684d46d40>, 'Result of Voronoi-Otsu-labeling (nsbatwm)', 2.0)
Result of Sum images (numpy, nsbatwm) <- (<function sum_images at 0x7f7684d47910>, 'slice_img', 'slice_img_c2', 2.0, -0.5)
Result of Voronoi-Otsu-labeling (nsbatwm) <- (<function voronoi_otsu_labeling at 0x7f7684d46ef0>, 'Result of Sum images (numpy, nsbatwm)', 15.0, 2.0)

wf.roots()=['slice_img', 'slice_img_c2']
wf.leafs()=['Result of Expand labels (scikit-image, nsbatwm)']
```

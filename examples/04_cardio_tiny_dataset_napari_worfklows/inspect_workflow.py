import sys

from napari_workflows._io_yaml_v1 import load_workflow

wf = load_workflow(sys.argv[1])

print(str(wf))
print(f"{wf.roots()=}")
print(f"{wf.leafs()=}")

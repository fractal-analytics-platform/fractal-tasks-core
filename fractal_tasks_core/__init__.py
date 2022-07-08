from .create_zarr_structure import create_zarr_structure  # noqa: F401
from .dummy import dummy  # noqa: F401

__FRACTAL_CORE_TASK_MANIFEST__ = [
    {
        "name": "dummy",
        "input_type": "Any",
        "output_type": "None",
    },
    {
        "name": "create_zarr_structure",
        "input_type": "image",
        "output_type": "zarr",
    },
]

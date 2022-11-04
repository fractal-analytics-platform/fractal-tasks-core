import logging
from typing import Optional

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s; %(levelname)s; %(message)s"
)


class MissingOptionalDependencyError(ModuleNotFoundError):
    def __init__(
        self, task: str, dependency: str, extra: Optional[str] = None,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.task = task
        self.dependency = dependency
        self.extra = extra or dependency

    def __str__(self):
        return (
            f"Task `{self.task}` depends on `{self.dependency}`, "
            "which does not appear to be installed. "
            f"Please install `fractal-tasks-core[{self.extra}]` "
            "to use this task."
        )


__VERSION__ = "0.2.6"
__OME_NGFF_VERSION__ = "0.4"

__FRACTAL_MANIFEST__ = [
    {
        "resource_type": "core task",
        "name": "Create OME-ZARR structure",
        "module": f"{__name__}.create_zarr_structure:create_zarr_structure",
        "input_type": "image",
        "output_type": "zarr",
        "default_args": {
            "num_levels": 2,
            "coarsening_xy": 2,
            "metadata_table": "mrf_mlf",
            "channel_parameters": None,
        },
    },
    {
        "name": "Yokogawa to Zarr",
        "resource_type": "core task",
        "input_type": "zarr",
        "output_type": "zarr",
        "module": f"{__name__}.yokogawa_to_zarr:yokogawa_to_zarr",
        "default_args": {"parallelization_level": "well"},
    },
    {
        "name": "Replicate Zarr structure",
        "resource_type": "core task",
        "input_type": "zarr",
        "output_type": "zarr",
        "module": f"{__name__}.replicate_zarr_structure:replicate_zarr_structure",  # noqa: E501
        "default_args": {
            "project_to_2D": True,
            "suffix": "mip",
        },
    },
    {
        "name": "Maximum Intensity Projection",
        "resource_type": "core task",
        "input_type": "zarr",
        "output_type": "zarr",
        "module": f"{__name__}.maximum_intensity_projection:maximum_intensity_projection",  # noqa: E501
        "default_args": {"parallelization_level": "well"},
    },
    {
        "name": "Cellpose Segmentation",
        "resource_type": "core task",
        "input_type": "zarr",
        "output_type": "zarr",
        "module": f"{__name__}.cellpose_segmentation:cellpose_segmentation",
        "default_args": {
            "labeling_channel": "A01_C01",
            "parallelization_level": "well",
        },
    },
    {
        "name": "Measurement",
        "resource_type": "core task",
        "input_type": "zarr",
        "output_type": "zarr",
        "module": f"{__name__}.measurement:measurement",
        "default_args": {
            "labeling_channel": "A01_C01",
            "level": 0,
            "measurement_table_name": "nuclei",
            "parallelization_level": "well",
        },
    },
    {
        "name": "Illumination correction",
        "resource_type": "core task",
        "input_type": "zarr",
        "output_type": "zarr",
        "module": f"{__name__}.illumination_correction:illumination_correction",  # noqa: E501
        "default_args": {
            "overwrite": False,
            "dict_corr": None,
            "background": 100,
            "parallelization_level": "well",
        },
    },
]

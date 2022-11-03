__VERSION__ = "0.1.5"
__OME_NGFF_VERSION__ = "0.4"


class MissingOptionalDependencyError(RuntimeError):
    pass


__FRACTAL_MANIFEST__ = [
    {
        "resource_type": "core task",
        "name": "dummy",
        "module": f"{__name__}.dummy:dummy",
        "input_type": "Any",
        "output_type": "None",
        "default_args": {
            "message": "dummy default",
            "index": 0,
        },
    },
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
        "name": "Per-FOV image labeling",
        "resource_type": "core task",
        "input_type": "zarr",
        "output_type": "zarr",
        "module": f"{__name__}.image_labeling:image_labeling",
        "default_args": {
            "labeling_channel": "A01_C01",
            "parallelization_level": "well",
        },
    },
    {
        "name": "Whole-well image labeling",
        "resource_type": "core task",
        "input_type": "zarr",
        "output_type": "zarr",
        "module": f"{__name__}.image_labeling_whole_well:image_labeling_whole_well",  # noqa: E501
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
            "table_name": "nuclei",
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

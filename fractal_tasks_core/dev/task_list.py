# Copyright 2022-2026 (C) BioVisionCenter, University of Zurich
"""
Fractal task list.
"""

from fractal_task_tools.task_models import (
    CompoundTask,
    ConverterNonParallelTask,
    ParallelTask,
)

AUTHORS = "Fractal Core Team"
DOCS_LINK = "https://fractal-analytics-platform.github.io/fractal-tasks-core"


TASK_LIST = [
    CompoundTask(
        name="Project Image (HCS Plate)",
        input_types={"is_3D": True},
        executable_init="init_projection_hcs.py",
        executable="compute_projection_hcs.py",
        output_types={"is_3D": False},
        meta_init={"cpus_per_task": 1, "mem": 4000},
        meta={"cpus_per_task": 1, "mem": 4000},
        category="Image Processing",
        modality="HCS",
        tags=["Preprocessing", "3D"],
        docs_info="file:task_info/projection_hcs.md",
    ),
    ParallelTask(
        name="Project Image",
        input_types={"is_3D": True},
        executable="projection.py",
        output_types={"is_3D": False},
        meta={"cpus_per_task": 1, "mem": 4000},
        category="Image Processing",
        tags=["Preprocessing", "3D"],
        docs_info="file:task_info/projection.md",
    ),
    ParallelTask(
        name="Illumination Correction",
        input_types=dict(illumination_corrected=False),
        executable="illumination_correction.py",
        output_types=dict(illumination_corrected=True),
        meta={"cpus_per_task": 1, "mem": 4000},
        category="Image Processing",
        tags=["Preprocessing", "2D", "3D"],
        docs_info="file:task_info/illumination_correction.md",
    ),
    ParallelTask(
        name="Threshold Segmentation",
        executable="threshold_segmentation.py",
        meta={"cpus_per_task": 1, "mem": 8000},
        category="Segmentation",
        tags=["2D", "3D", "Segmentation", "Otsu"],
        docs_info="file:task_info/threshold_segmentation.md",
    ),
    ParallelTask(
        name="Measure Features",
        executable="measure_features.py",
        meta={"cpus_per_task": 1, "mem": 4000},
        category="Measurement",
        tags=["2D", "3D", "Feature Extraction"],
        docs_info="file:task_info/measure_features.md",
    ),
    CompoundTask(
        name="Calculate Registration (image-based)",
        executable_init="init_image_based_registration.py",
        executable="compute_image_based_registration.py",
        meta_init={"cpus_per_task": 1, "mem": 1000},
        meta={"cpus_per_task": 1, "mem": 8000},
        category="Registration",
        modality="HCS",
        tags=["Multiplexing", "2D", "3D"],
        docs_info="file:task_info/image_based_registration.md",
    ),
    CompoundTask(
        name="Find Registration Consensus",
        executable_init="init_registration_consensus.py",
        executable="compute_registration_consensus.py",
        meta_init={"cpus_per_task": 1, "mem": 1000},
        meta={"cpus_per_task": 1, "mem": 1000},
        category="Registration",
        modality="HCS",
        tags=["Multiplexing", "2D", "3D"],
        docs_info="file:task_info/registration_consensus.md",
    ),
    ParallelTask(
        name="Apply Registration to Image",
        input_types=dict(registered=False),
        executable="apply_registration_to_image.py",
        output_types=dict(registered=True),
        meta={"cpus_per_task": 1, "mem": 4000},
        category="Registration",
        modality="HCS",
        tags=["Multiplexing", "2D", "3D"],
        docs_info="file:task_info/apply_registration_to_image.md",
    ),
    ConverterNonParallelTask(
        name="Import OME-Zarr",
        executable="import_ome_zarr.py",
        docs_info="file:task_info/import_ome_zarr.md",
        tags=["2D", "3D"],
    ),
]

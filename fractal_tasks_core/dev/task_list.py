# Copyright 2024 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Joel Lüthi <joel.luethi@uzh.ch>
#
# This file is part of Fractal and was originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
"""
Fractal task list.
"""
from fractal_task_tools.task_models import CompoundTask
from fractal_task_tools.task_models import ConverterCompoundTask
from fractal_task_tools.task_models import ConverterNonParallelTask
from fractal_task_tools.task_models import ParallelTask

AUTHORS = "Fractal Core Team"
DOCS_LINK = "https://fractal-analytics-platform.github.io/fractal-tasks-core"
INPUT_MODELS = [
    ["fractal_tasks_core", "channels.py", "OmeroChannel"],
    ["fractal_tasks_core", "channels.py", "Window"],
    ["fractal_tasks_core", "channels.py", "ChannelInputModel"],
    ["fractal_tasks_core", "tasks/io_models.py", "NapariWorkflowsInput"],
    ["fractal_tasks_core", "tasks/io_models.py", "NapariWorkflowsOutput"],
    [
        "fractal_tasks_core",
        "tasks/cellpose_utils.py",
        "CellposeCustomNormalizer",
    ],
    [
        "fractal_tasks_core",
        "tasks/cellpose_utils.py",
        "CellposeChannel1InputModel",
    ],
    [
        "fractal_tasks_core",
        "tasks/cellpose_utils.py",
        "CellposeChannel2InputModel",
    ],
    ["fractal_tasks_core", "tasks/cellpose_utils.py", "CellposeModelParams"],
    ["fractal_tasks_core", "tasks/io_models.py", "MultiplexingAcquisition"],
]


TASK_LIST = [
    ConverterCompoundTask(
        name="Convert Cellvoyager to OME-Zarr",
        executable_init="tasks/cellvoyager_to_ome_zarr_init.py",
        executable="tasks/cellvoyager_to_ome_zarr_compute.py",
        meta_init={"cpus_per_task": 1, "mem": 4000},
        meta={"cpus_per_task": 1, "mem": 4000},
        category="Conversion",
        modality="HCS",
        tags=["Yokogawa", "Cellvoyager", "2D", "3D"],
        docs_info="file:task_info/convert_cellvoyager_to_ome_zarr.md",
    ),
    ConverterCompoundTask(
        name="Convert Cellvoyager Multiplexing to OME-Zarr",
        executable_init="tasks/cellvoyager_to_ome_zarr_init_multiplex.py",
        executable="tasks/cellvoyager_to_ome_zarr_compute.py",
        meta_init={"cpus_per_task": 1, "mem": 4000},
        meta={"cpus_per_task": 1, "mem": 4000},
        category="Conversion",
        modality="HCS",
        tags=["Yokogawa", "Cellvoyager", "2D", "3D"],
        docs_info="file:task_info/convert_cellvoyager_multiplex.md",
    ),
    CompoundTask(
        name="Project Image (HCS Plate)",
        input_types={"is_3D": True},
        executable_init="tasks/copy_ome_zarr_hcs_plate.py",
        executable="tasks/projection.py",
        output_types={"is_3D": False},
        meta_init={"cpus_per_task": 1, "mem": 4000},
        meta={"cpus_per_task": 1, "mem": 4000},
        category="Image Processing",
        modality="HCS",
        tags=["Preprocessing", "3D"],
        docs_info="file:task_info/projection.md",
    ),
    ParallelTask(
        name="Illumination Correction",
        input_types=dict(illumination_corrected=False),
        executable="tasks/illumination_correction.py",
        output_types=dict(illumination_corrected=True),
        meta={"cpus_per_task": 1, "mem": 4000},
        category="Image Processing",
        tags=["Preprocessing", "2D", "3D"],
        docs_info="file:task_info/illumination_correction.md",
    ),
    ParallelTask(
        name="Cellpose Segmentation",
        executable="tasks/cellpose_segmentation.py",
        meta={"cpus_per_task": 4, "mem": 16000, "needs_gpu": True},
        category="Segmentation",
        tags=[
            "Deep Learning",
            "Convolutional Neural Network",
            "Instance Segmentation",
            "2D",
            "3D",
        ],
        docs_info="file:task_info/cellpose_segmentation.md",
    ),
    CompoundTask(
        name="Calculate Registration (image-based)",
        executable_init="tasks/image_based_registration_hcs_init.py",
        executable="tasks/calculate_registration_image_based.py",
        meta_init={"cpus_per_task": 1, "mem": 1000},
        meta={"cpus_per_task": 1, "mem": 8000},
        category="Registration",
        modality="HCS",
        tags=["Multiplexing", "2D", "3D"],
        docs_info="file:task_info/calculate_registration_image_based.md",
    ),
    CompoundTask(
        name="Find Registration Consensus",
        executable_init="tasks/init_group_by_well_for_multiplexing.py",
        executable="tasks/find_registration_consensus.py",
        meta_init={"cpus_per_task": 1, "mem": 1000},
        meta={"cpus_per_task": 1, "mem": 1000},
        category="Registration",
        modality="HCS",
        tags=["Multiplexing", "2D", "3D"],
        docs_info="file:task_info/find_registration_consensus.md",
    ),
    ParallelTask(
        name="Apply Registration to Image",
        input_types=dict(registered=False),
        executable="tasks/apply_registration_to_image.py",
        output_types=dict(registered=True),
        meta={"cpus_per_task": 1, "mem": 4000},
        category="Registration",
        modality="HCS",
        tags=["Multiplexing", "2D", "3D"],
        docs_info="file:task_info/apply_registration_to_image.md",
    ),
    ConverterNonParallelTask(
        name="Import OME-Zarr",
        executable="tasks/import_ome_zarr.py",
        docs_info="file:task_info/import_ome_zarr.md",
        tags=["2D", "3D"],
    ),
    ParallelTask(
        name="Napari Workflows Wrapper",
        executable="tasks/napari_workflows_wrapper.py",
        meta={
            "cpus_per_task": 8,
            "mem": 32000,
        },
        category="Measurement",
        tags=["2D", "3D"],
        docs_info="file:task_info/napari_workflows_wrapper.md",
    ),
]

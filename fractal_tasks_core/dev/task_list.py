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
from fractal_tasks_core.dev.task_models import CompoundTask
from fractal_tasks_core.dev.task_models import NonParallelTask
from fractal_tasks_core.dev.task_models import ParallelTask

TASK_LIST = [
    CompoundTask(
        name="Convert Cellvoyager to OME-Zarr",
        executable_init="tasks/cellvoyager_to_ome_zarr_init.py",
        executable="tasks/cellvoyager_to_ome_zarr_compute.py",
        meta_init={"cpus_per_task": 1, "mem": 4000},
        meta={"cpus_per_task": 1, "mem": 4000},
    ),
    CompoundTask(
        name="Convert Cellvoyager Multiplexing to OME-Zarr",
        executable_init="tasks/cellvoyager_to_ome_zarr_init_multiplex.py",
        executable="tasks/cellvoyager_to_ome_zarr_compute.py",
        meta_init={"cpus_per_task": 1, "mem": 4000},
        meta={"cpus_per_task": 1, "mem": 4000},
    ),
    CompoundTask(
        name="Maximum Intensity Projection HCS Plate",
        input_types={"is_3D": True},
        executable_init="tasks/copy_ome_zarr_hcs_plate.py",
        executable="tasks/maximum_intensity_projection.py",
        output_types={"is_3D": False},
        meta_init={"cpus_per_task": 1, "mem": 4000},
        meta={"cpus_per_task": 1, "mem": 4000},
    ),
    ParallelTask(
        name="Illumination Correction",
        input_types=dict(illumination_corrected=False),
        executable="tasks/illumination_correction.py",
        output_types=dict(illumination_corrected=True),
        meta={"cpus_per_task": 1, "mem": 4000},
    ),
    # CompoundTask(
    #     name="Illumination Correction (channel parallelized)",
    #     input_types=dict(illumination_correction=False),
    #     executable_init="illumination_correction_init.py",
    #     executable="illumination_correction_compute.py",
    #     output_types=dict(illumination_correction=True),
    # ),
    ParallelTask(
        name="Cellpose Segmentation",
        executable="tasks/cellpose_segmentation.py",
        meta={"cpus_per_task": 4, "mem": 16000, "needs_gpu": True},
    ),
    CompoundTask(
        name="Calculate Registration (image-based)",
        executable_init="tasks/image_based_registration_hcs_init.py",
        executable="tasks/calculate_registration_image_based.py",
        meta_init={"cpus_per_task": 1, "mem": 1000},
        meta={"cpus_per_task": 1, "mem": 8000},
    ),
    CompoundTask(
        name="Find Registration Consensus",
        executable_init="tasks/init_group_by_well_for_multiplexing.py",
        executable="tasks/find_registration_consensus.py",
        meta_init={"cpus_per_task": 1, "mem": 1000},
        meta={"cpus_per_task": 1, "mem": 1000},
    ),
    ParallelTask(
        name="Apply Registration to Image",
        input_types=dict(registered=False),
        executable="tasks/apply_registration_to_image.py",
        output_types=dict(registered=True),
        meta={"cpus_per_task": 1, "mem": 4000},
    ),
    NonParallelTask(
        name="Import OME-Zarr",
        executable="tasks/import_ome_zarr.py",
    ),
    ParallelTask(
        name="Napari Workflows Wrapper",
        executable="tasks/napari_workflows_wrapper.py",
        meta={
            "cpus_per_task": 8,
            "mem": 32000,
        },
    ),
]

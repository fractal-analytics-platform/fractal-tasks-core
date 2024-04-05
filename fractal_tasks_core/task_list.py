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
from fractal_tasks_core.dev.task_models import ParallelTask

# from fractal_tasks_core.dev.task_models import NonParallelTask

TASK_LIST = [
    CompoundTask(
        name="Convert Cellvoyager to OME-Zarr",
        executable_init="cellvoyager_to_ome_zarr_init.py",
        executable="cellvoyager_to_ome_zarr_compute.py",
        meta_init={"cpus_per_task": 1, "mem": 4000},
        meta={"cpus_per_task": 1, "mem": 4000},
    ),
    CompoundTask(
        name="Convert Cellvoyager Multiplexing to OME-Zarr",
        executable_init="cellvoyager_to_ome_zarr_init_multiplex.py",
        executable="cellvoyager_to_ome_zarr_compute.py",
        meta_init={"cpus_per_task": 1, "mem": 4000},
        meta={"cpus_per_task": 1, "mem": 4000},
    ),
    CompoundTask(
        name="Maximum Intensity Projection HCS Plate",
        input_types={"3D": True},
        executable_init="copy_ome_zarr_hcs_plate.py",
        executable="maximum_intensity_projection.py",
        output_types={"3D": False},
        meta_init={"cpus_per_task": 1, "mem": 4000},
        meta={"cpus_per_task": 1, "mem": 4000},
    ),
    ParallelTask(
        name="Illumination Correction",
        input_types=dict(illumination_corrected=False),
        executable="illumination_correction.py",
        output_types=dict(illumination_corrected=True),
        meta={"cpus_per_task": 1, "mem": 4000},
    ),
    # CompoundTask(
    #     name="illumination_correction_compound",
    #     input_types=dict(illumination_correction=False),
    #     executable_init="illumination_correction_init.py",
    #     executable="illumination_correction_compute.py",
    #     output_types=dict(illumination_correction=True),
    # ),
    ParallelTask(
        name="cellpose_segmentation",
        executable="cellpose_segmentation.py",
        meta={"cpus_per_task": 4, "mem": 16000, "needs_gpu": True},
    ),
    # CompoundTask(
    #     name="calculate_registration_compound",
    #     executable_init="calculate_registration_init.py",
    #     executable="calculate_registration_compute.py",
    # ),
    # NonParallelTask(
    #     name="find_registration_consensus",
    #     executable="find_registration_consensus.py",
    # ),
    ParallelTask(
        name="Apply Registration to Image",
        input_types=dict(registered=False),
        executable="apply_registration_to_image.py",
        output_types=dict(registered=True),
        meta={"cpus_per_task": 1, "mem": 4000},
    ),
]

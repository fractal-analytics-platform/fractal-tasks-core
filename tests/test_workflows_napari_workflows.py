"""
Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
University of Zurich

Original authors:
Marco Franzon <marco.franzon@exact-lab.it>
Tommaso Comparin <tommaso.comparin@exact-lab.it>

This file is part of Fractal and was originally developed by eXact lab S.r.l.
<exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
Institute for Biomedical Research and Pelkmans Lab from the University of
Zurich.
"""
import shutil
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List

from devtools import debug
from utils import check_file_number
from utils import validate_labels_and_measurements
from utils import validate_schema

from fractal_tasks_core.napari_workflows_wrapper import (
    napari_workflows_wrapper,
)


def prepare_3D_zarr(
    zarr_path: Path,
    zenodo_zarr: List[Path],
    zenodo_zarr_metadata: List[Dict[str, Any]],
):
    zenodo_zarr_3D, zenodo_zarr_2D = zenodo_zarr[:]
    metadata_3D, metadata_2D = zenodo_zarr_metadata[:]
    shutil.copytree(
        str(zenodo_zarr_3D), str(zarr_path.parent / zenodo_zarr_3D.name)
    )
    metadata = metadata_3D.copy()
    return metadata


def test_workflow_napari_worfklow(
    tmp_path: Path,
    testdata_path: Path,
    zenodo_zarr: List[Path],
    zenodo_zarr_metadata: List[Dict[str, Any]],
):

    # Init
    zarr_path = tmp_path / "tmp_out/*.zarr"
    metadata = prepare_3D_zarr(zarr_path, zenodo_zarr, zenodo_zarr_metadata)
    debug(zarr_path)
    debug(metadata)

    # First napari-workflows task (labeling)
    workflow_file = str(testdata_path / "napari_workflows/wf_1.yaml")
    input_specs = {
        "input": {"type": "image", "channel": "A01_C01"},
    }
    output_specs = {
        "Result of Expand labels (scikit-image, nsbatwm)": {
            "type": "label",
            "label_name": "label_DAPI",
        },
    }
    for component in metadata["well"]:
        napari_workflows_wrapper(
            input_paths=[zarr_path],
            output_path=zarr_path,
            metadata=metadata,
            component=component,
            input_specs=input_specs,
            output_specs=output_specs,
            workflow_file=workflow_file,
            ROI_table_name="FOV_ROI_table",
            level=2,
        )
    debug(metadata)

    # Second napari-workflows task (measurement)
    workflow_file = str(testdata_path / "napari_workflows/wf_4.yaml")
    input_specs = {
        "dapi_img": {"type": "image", "channel": "A01_C01"},
        "dapi_label_img": {"type": "label", "label_name": "label_DAPI"},
    }
    output_specs = {
        "regionprops_DAPI": {
            "type": "dataframe",
            "table_name": "regionprops_DAPI",
        },
    }
    for component in metadata["well"]:
        napari_workflows_wrapper(
            input_paths=[zarr_path],
            output_path=zarr_path,
            metadata=metadata,
            component=component,
            input_specs=input_specs,
            output_specs=output_specs,
            workflow_file=workflow_file,
            ROI_table_name="FOV_ROI_table",
        )
    debug(metadata)

    # OME-NGFF JSON validation
    image_zarr = Path(zarr_path.parent / metadata["well"][0])
    well_zarr = image_zarr.parent
    plate_zarr = image_zarr.parents[2]
    label_zarr = image_zarr / "labels/label_DAPI"
    validate_schema(path=str(image_zarr), type="image")
    validate_schema(path=str(well_zarr), type="well")
    validate_schema(path=str(plate_zarr), type="plate")
    validate_schema(path=str(label_zarr), type="label")

    check_file_number(zarr_path=image_zarr)

    validate_labels_and_measurements(
        image_zarr, label_name="label_DAPI", table_name="regionprops_DAPI"
    )


def test_workflow_napari_worfklow_label_input_only(
    tmp_path: Path,
    testdata_path: Path,
    zenodo_zarr: List[Path],
    zenodo_zarr_metadata: List[Dict[str, Any]],
):

    # Init
    zarr_path = tmp_path / "tmp_out/*.zarr"
    metadata = prepare_3D_zarr(zarr_path, zenodo_zarr, zenodo_zarr_metadata)
    debug(zarr_path)
    debug(metadata)

    # First napari-workflows task (labeling)
    workflow_file = str(testdata_path / "napari_workflows/wf_1.yaml")
    input_specs = {
        "input": {"type": "image", "channel": "A01_C01"},
    }
    output_specs = {
        "Result of Expand labels (scikit-image, nsbatwm)": {
            "type": "label",
            "label_name": "label_DAPI",
        },
    }
    for component in metadata["well"]:
        napari_workflows_wrapper(
            input_paths=[zarr_path],
            output_path=zarr_path,
            metadata=metadata,
            component=component,
            input_specs=input_specs,
            output_specs=output_specs,
            workflow_file=workflow_file,
            ROI_table_name="FOV_ROI_table",
            level=2,
        )
    debug(metadata)

    # Second napari-workflows task (measurement)
    workflow_file = str(
        testdata_path / "napari_workflows" / "wf_from_labels_to_labels.yaml"
    )
    input_specs = {
        "test_labels": {"type": "label", "label_name": "label_DAPI"},
    }
    output_specs = {
        "Result of Expand labels (scikit-image, nsbatwm)": {
            "type": "label",
            "label_name": "label_DAPI_expanded",
        },
    }
    for component in metadata["well"]:
        napari_workflows_wrapper(
            input_paths=[zarr_path],
            output_path=zarr_path,
            metadata=metadata,
            component=component,
            input_specs=input_specs,
            output_specs=output_specs,
            workflow_file=workflow_file,
            ROI_table_name="FOV_ROI_table",
        )
    debug(metadata)

    # OME-NGFF JSON validation
    image_zarr = Path(zarr_path.parent / metadata["well"][0])
    well_zarr = image_zarr.parent
    plate_zarr = image_zarr.parents[2]
    label_zarr = image_zarr / "labels/label_DAPI"
    validate_schema(path=str(image_zarr), type="image")
    validate_schema(path=str(well_zarr), type="well")
    validate_schema(path=str(plate_zarr), type="plate")
    validate_schema(path=str(label_zarr), type="label")

    check_file_number(zarr_path=image_zarr)

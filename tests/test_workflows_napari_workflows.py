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
from typing import Union

import anndata as ad
import pytest
from devtools import debug

from .utils import check_file_number
from .utils import validate_labels_and_measurements
from .utils import validate_schema
from fractal_tasks_core.napari_workflows_wrapper import (
    napari_workflows_wrapper,
)


def prepare_3D_zarr(
    zarr_path: str,
    zenodo_zarr: List[str],
    zenodo_zarr_metadata: List[Dict[str, Any]],
):
    zenodo_zarr_3D, zenodo_zarr_2D = zenodo_zarr[:]
    metadata_3D, metadata_2D = zenodo_zarr_metadata[:]
    shutil.copytree(
        zenodo_zarr_3D, str(Path(zarr_path).parent / Path(zenodo_zarr_3D).name)
    )
    metadata = metadata_3D.copy()
    return metadata


def prepare_2D_zarr(
    zarr_path: str,
    zenodo_zarr: List[str],
    zenodo_zarr_metadata: List[Dict[str, Any]],
    remove_labels: bool = False,
):
    zenodo_zarr_3D, zenodo_zarr_2D = zenodo_zarr[:]
    metadata_3D, metadata_2D = zenodo_zarr_metadata[:]
    shutil.copytree(
        zenodo_zarr_2D, str(Path(zarr_path).parent / Path(zenodo_zarr_2D).name)
    )
    if remove_labels:
        label_dir = str(
            Path(zarr_path).parent
            / Path(zenodo_zarr_2D).name
            / "B/03/0/labels"
        )
        debug(label_dir)
        shutil.rmtree(label_dir)
    metadata = metadata_2D.copy()
    return metadata


def test_napari_worfklow(
    tmp_path: Path,
    testdata_path: Path,
    zenodo_zarr: List[str],
    zenodo_zarr_metadata: List[Dict[str, Any]],
):

    # Init
    zarr_path = tmp_path / "tmp_out/*.zarr"
    metadata = prepare_3D_zarr(
        str(zarr_path), zenodo_zarr, zenodo_zarr_metadata
    )
    debug(zarr_path)
    debug(metadata)

    # First napari-workflows task (labeling)
    workflow_file = str(testdata_path / "napari_workflows/wf_1.yaml")
    input_specs: Dict[str, Dict[str, Union[str, int]]] = {
        "input": {"type": "image", "wavelength_id": "A01_C01"},
    }
    output_specs: Dict[str, Dict[str, Union[str, int]]] = {
        "Result of Expand labels (scikit-image, nsbatwm)": {
            "type": "label",
            "label_name": "label_DAPI",
        },
    }
    for component in metadata["image"]:
        napari_workflows_wrapper(
            input_paths=[str(zarr_path)],
            output_path=str(zarr_path),
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
        "dapi_img": {"type": "image", "wavelength_id": "A01_C01"},
        "dapi_label_img": {"type": "label", "label_name": "label_DAPI"},
    }
    output_specs = {
        "regionprops_DAPI": {
            "type": "dataframe",
            "table_name": "regionprops_DAPI",
        },
    }
    for component in metadata["image"]:
        napari_workflows_wrapper(
            input_paths=[str(zarr_path)],
            output_path=str(zarr_path),
            metadata=metadata,
            component=component,
            input_specs=input_specs,
            output_specs=output_specs,
            workflow_file=workflow_file,
            ROI_table_name="FOV_ROI_table",
        )
    debug(metadata)

    # OME-NGFF JSON validation
    image_zarr = zarr_path.parent / metadata["image"][0]
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

    # Load measurements
    meas = ad.read_zarr(
        str(
            zarr_path.parent
            / metadata["image"][0]
            / "tables/regionprops_DAPI/"
        )
    )
    debug(meas.var_names)
    assert "area" in meas.var_names
    assert "bbox_area" in meas.var_names


def test_napari_worfklow_label_input_only(
    tmp_path: Path,
    testdata_path: Path,
    zenodo_zarr: List[str],
    zenodo_zarr_metadata: List[Dict[str, Any]],
):

    # Prepare 3D zarr
    zarr_path = tmp_path / "tmp_out/*.zarr"
    metadata = prepare_3D_zarr(
        str(zarr_path), zenodo_zarr, zenodo_zarr_metadata
    )
    debug(zarr_path)
    debug(metadata)

    # First napari-workflows task (labeling)
    workflow_file = str(testdata_path / "napari_workflows/wf_1.yaml")
    input_specs: Dict[str, Dict[str, Union[str, int]]] = {
        "input": {"type": "image", "wavelength_id": "A01_C01"},
    }
    output_specs: Dict[str, Dict[str, Union[str, int]]] = {
        "Result of Expand labels (scikit-image, nsbatwm)": {
            "type": "label",
            "label_name": "label_DAPI",
        },
    }
    for component in metadata["image"]:
        napari_workflows_wrapper(
            input_paths=[str(zarr_path)],
            output_path=str(zarr_path),
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
    for component in metadata["image"]:
        napari_workflows_wrapper(
            input_paths=[str(zarr_path)],
            output_path=str(zarr_path),
            metadata=metadata,
            component=component,
            input_specs=input_specs,
            output_specs=output_specs,
            workflow_file=workflow_file,
            ROI_table_name="FOV_ROI_table",
        )
    debug(metadata)

    # OME-NGFF JSON validation
    image_zarr = zarr_path.parent / metadata["image"][0]
    well_zarr = image_zarr.parent
    plate_zarr = image_zarr.parents[2]
    label_zarr = image_zarr / "labels/label_DAPI"
    validate_schema(path=str(image_zarr), type="image")
    validate_schema(path=str(well_zarr), type="well")
    validate_schema(path=str(plate_zarr), type="plate")
    validate_schema(path=str(label_zarr), type="label")

    check_file_number(zarr_path=image_zarr)


# Define three relabeling scenarios:
LABEL_NAME = "label_DAPI"
TABLE_NAME = "measurement_DAPI"
# 1. Labeling-only workflow, from images to labels.
workflow_file_name = "wf_relab_1-labeling_only.yaml"
input_specs = dict(input_image={"type": "image", "wavelength_id": "A01_C01"})
output_specs = dict(output_label={"type": "label", "label_name": LABEL_NAME})
RELABELING_CASE_1: List = [workflow_file_name, input_specs, output_specs]
# 2. Measurement-only workflow, from images+labels to dataframes.
workflow_file_name = "wf_relab_2-measurement_only.yaml"
input_specs = dict(
    input_image={"type": "image", "wavelength_id": "A01_C01"},
    input_label={"type": "label", "label_name": LABEL_NAME},
)
output_specs = dict(
    output_dataframe={"type": "dataframe", "table_name": TABLE_NAME}
)
RELABELING_CASE_2: List = [workflow_file_name, input_specs, output_specs]
# 3. Mixed labeling/measurement workflow.
workflow_file_name = "wf_relab_3-labeling_and_measurement.yaml"
input_specs = dict(input_image={"type": "image", "wavelength_id": "A01_C01"})
output_specs = dict(
    output_label={"type": "label", "label_name": LABEL_NAME},
    output_dataframe={"type": "dataframe", "table_name": TABLE_NAME},
)
RELABELING_CASE_3: List = [workflow_file_name, input_specs, output_specs]
# Assemble three cases
relabeling_cases = []
relabeling_cases.append(RELABELING_CASE_1 + [False])
relabeling_cases.append(RELABELING_CASE_2 + [True])
relabeling_cases.append(RELABELING_CASE_3 + [False])


@pytest.mark.parametrize(
    "workflow_file_name,input_specs,output_specs,needs_labels",
    relabeling_cases,
)
def test_relabeling(
    workflow_file_name: str,
    input_specs: Dict[str, Dict],
    output_specs: Dict[str, Dict],
    needs_labels: bool,
    tmp_path: Path,
    testdata_path: Path,
    zenodo_zarr: List[str],
    zenodo_zarr_metadata: List[Dict[str, Any]],
):

    # Prepare 3D zarr
    zarr_path = tmp_path / "tmp_out/*.zarr"
    metadata = prepare_3D_zarr(
        str(zarr_path), zenodo_zarr, zenodo_zarr_metadata
    )
    debug(zarr_path)
    debug(metadata)

    # If needed, produce some labels before the actual test
    if needs_labels:
        workflow_file = str(
            testdata_path / "napari_workflows" / RELABELING_CASE_1[0]
        )
        for component in metadata["image"]:
            napari_workflows_wrapper(
                input_paths=[str(zarr_path)],
                output_path=str(zarr_path),
                metadata=metadata,
                component=component,
                input_specs=RELABELING_CASE_1[1],
                output_specs=RELABELING_CASE_1[2],
                workflow_file=workflow_file,
                ROI_table_name="FOV_ROI_table",
            )

    # Run napari-workflow
    workflow_file = str(
        testdata_path / "napari_workflows" / workflow_file_name
    )
    debug(workflow_file)
    debug(input_specs)
    debug(output_specs)
    for component in metadata["image"]:
        napari_workflows_wrapper(
            input_paths=[str(zarr_path)],
            output_path=str(zarr_path),
            metadata=metadata,
            component=component,
            input_specs=input_specs,
            output_specs=output_specs,
            workflow_file=workflow_file,
            ROI_table_name="FOV_ROI_table",
        )
    debug(metadata)

    image_zarr = Path(zarr_path.parent / metadata["image"][0])
    validate_labels_and_measurements(
        image_zarr, label_name=LABEL_NAME, table_name=TABLE_NAME
    )

    dataframe_outputs = [
        item for item in output_specs.values() if item["type"] == "dataframe"
    ]
    if dataframe_outputs:
        meas = ad.read_zarr(
            zarr_path.parent / metadata["image"][0] / f"tables/{TABLE_NAME}/"
        )
        debug(meas.var_names)
        assert "area" in meas.var_names
        assert "bbox_area" in meas.var_names


def test_fail_if_no_relabeling(
    tmp_path: Path,
    testdata_path: Path,
    zenodo_zarr: List[str],
    zenodo_zarr_metadata: List[Dict[str, Any]],
):

    # Prepare 3D zarr
    zarr_path = tmp_path / "tmp_out/*.zarr"
    metadata = prepare_3D_zarr(
        str(zarr_path), zenodo_zarr, zenodo_zarr_metadata
    )
    debug(zarr_path)
    debug(metadata)

    # Run napari-workflow RELABELING_CASE_1, but with relabeling=False
    workflow_file_name, input_specs, output_specs = RELABELING_CASE_1
    workflow_file = str(
        testdata_path / "napari_workflows" / workflow_file_name
    )
    debug(workflow_file)
    debug(input_specs)
    debug(output_specs)
    for component in metadata["image"]:
        napari_workflows_wrapper(
            input_paths=[str(zarr_path)],
            output_path=str(zarr_path),
            metadata=metadata,
            component=component,
            input_specs=input_specs,
            output_specs=output_specs,
            workflow_file=workflow_file,
            ROI_table_name="FOV_ROI_table",
            relabeling=False,
        )
    debug(metadata)

    image_zarr = zarr_path.parent / metadata["image"][0]
    with pytest.raises(AssertionError):
        validate_labels_and_measurements(
            image_zarr, label_name=LABEL_NAME, table_name=TABLE_NAME
        )


cases = [
    (2, 2, True),
    (2, 3, False),
    (3, 3, True),
    (3, 2, True),
]


@pytest.mark.parametrize(
    "expected_dimensions,zarr_dimensions,expected_success", cases
)
def test_expected_dimensions(
    expected_dimensions: int,
    zarr_dimensions: int,
    expected_success: bool,
    tmp_path: Path,
    testdata_path: Path,
    zenodo_zarr: List[str],
    zenodo_zarr_metadata: List[Dict[str, Any]],
):

    # Prepare zarr
    zarr_path = tmp_path / "tmp_out/*.zarr"
    if zarr_dimensions == 2:
        metadata = prepare_2D_zarr(
            str(zarr_path),
            zenodo_zarr,
            zenodo_zarr_metadata,
            remove_labels=True,
        )
    else:
        metadata = prepare_3D_zarr(
            str(zarr_path), zenodo_zarr, zenodo_zarr_metadata
        )
    debug(zarr_path)
    debug(metadata)

    # First napari-workflows task (labeling)
    workflow_file = str(
        testdata_path / "napari_workflows/wf_5-labeling_only.yaml"
    )
    input_specs: Dict[str, Dict[str, Union[str, int]]] = {
        "input_image": {"type": "image", "wavelength_id": "A01_C01"},
    }
    output_specs: Dict[str, Dict[str, Union[str, int]]] = {
        "output_label": {
            "type": "label",
            "label_name": "label_DAPI",
        },
    }

    for component in metadata["image"]:
        arguments = dict(
            input_paths=[str(zarr_path)],
            output_path=str(zarr_path),
            metadata=metadata,
            component=component,
            input_specs=input_specs,
            output_specs=output_specs,
            workflow_file=workflow_file,
            ROI_table_name="FOV_ROI_table",
            level=3,
            expected_dimensions=expected_dimensions,
        )
        if expected_success:
            napari_workflows_wrapper(**arguments)
        else:
            with pytest.raises(ValueError):
                napari_workflows_wrapper(**arguments)

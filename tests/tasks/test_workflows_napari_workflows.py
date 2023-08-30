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
from pathlib import Path
from typing import Any

import anndata as ad
import pytest
from devtools import debug

from .._zenodo_ome_zarrs import prepare_2D_zarr
from .._zenodo_ome_zarrs import prepare_3D_zarr
from ._validation import check_file_number
from ._validation import validate_axes_and_coordinateTransformations
from ._validation import validate_labels_and_measurements
from ._validation import validate_schema
from .lib_empty_ROI_table import _add_empty_ROI_table
from fractal_tasks_core.lib_input_models import NapariWorkflowsInput
from fractal_tasks_core.lib_input_models import NapariWorkflowsOutput
from fractal_tasks_core.lib_zarr import OverwriteNotAllowedError
from fractal_tasks_core.tasks.napari_workflows_wrapper import (
    napari_workflows_wrapper,
)

try:
    import napari_skimage_regionprops_mock

    has_napari_skimage_regionprops_mock = True
    print(napari_skimage_regionprops_mock)
except ModuleNotFoundError:
    has_napari_skimage_regionprops_mock = False


def test_napari_workflow(
    tmp_path: Path,
    testdata_path: Path,
    zenodo_zarr: list[str],
    zenodo_zarr_metadata: list[dict[str, Any]],
):

    # Init
    zarr_path = tmp_path / "tmp_out/"
    metadata = prepare_3D_zarr(
        str(zarr_path), zenodo_zarr, zenodo_zarr_metadata
    )
    debug(zarr_path)
    debug(metadata)

    # Prepare parameters for first napari-workflows task (labeling)
    workflow_file = str(testdata_path / "napari_workflows/wf_1.yaml")
    input_specs: dict[str, NapariWorkflowsInput] = {
        "input": {"type": "image", "channel": {"wavelength_id": "A01_C01"}},  # type: ignore # noqa
    }
    output_specs: dict[str, NapariWorkflowsOutput] = {
        "Result of Expand labels (scikit-image, nsbatwm)": {  # type: ignore # noqa
            "type": "label",
            "label_name": "label_DAPI",
        },
    }

    # Run once
    for component in metadata["image"]:
        napari_workflows_wrapper(
            input_paths=[str(zarr_path)],
            output_path=str(zarr_path),
            metadata=metadata,
            component=component,
            input_specs=input_specs,
            output_specs=output_specs,
            workflow_file=workflow_file,
            input_ROI_table="FOV_ROI_table",
            level=2,
        )
    debug(metadata)

    # Re-run with overwrite=True
    for component in metadata["image"]:
        napari_workflows_wrapper(
            input_paths=[str(zarr_path)],
            output_path=str(zarr_path),
            metadata=metadata,
            component=component,
            input_specs=input_specs,
            output_specs=output_specs,
            workflow_file=workflow_file,
            input_ROI_table="FOV_ROI_table",
            level=2,
            overwrite=True,
        )

    # Re-run with overwrite=False
    with pytest.raises(Exception):
        for component in metadata["image"]:
            napari_workflows_wrapper(
                input_paths=[str(zarr_path)],
                output_path=str(zarr_path),
                metadata=metadata,
                component=component,
                input_specs=input_specs,
                output_specs=output_specs,
                workflow_file=workflow_file,
                input_ROI_table="FOV_ROI_table",
                level=2,
                overwrite=False,
            )

    # Prepare parameters for second napari-workflows task (measurement)
    workflow_file = str(testdata_path / "napari_workflows/wf_4.yaml")
    input_specs = {
        "dapi_img": {"type": "image", "channel": {"wavelength_id": "A01_C01"}},  # type: ignore # noqa
        "dapi_label_img": {"type": "label", "label_name": "label_DAPI"},  # type: ignore # noqa
    }
    output_specs = {
        "regionprops_DAPI": {  # type: ignore # noqa
            "type": "dataframe",
            "table_name": "regionprops_DAPI",
        },
    }

    # Run once
    for component in metadata["image"]:
        napari_workflows_wrapper(
            input_paths=[str(zarr_path)],
            output_path=str(zarr_path),
            metadata=metadata,
            component=component,
            input_specs=input_specs,
            output_specs=output_specs,
            workflow_file=workflow_file,
            input_ROI_table="FOV_ROI_table",
        )
    debug(metadata)

    # Re-run with overwrite=True
    for component in metadata["image"]:
        napari_workflows_wrapper(
            input_paths=[str(zarr_path)],
            output_path=str(zarr_path),
            metadata=metadata,
            component=component,
            input_specs=input_specs,
            output_specs=output_specs,
            workflow_file=workflow_file,
            input_ROI_table="FOV_ROI_table",
            overwrite=True,
        )

    # Re-run with overwrite=False
    with pytest.raises(OverwriteNotAllowedError):
        for component in metadata["image"]:
            napari_workflows_wrapper(
                input_paths=[str(zarr_path)],
                output_path=str(zarr_path),
                metadata=metadata,
                component=component,
                input_specs=input_specs,
                output_specs=output_specs,
                workflow_file=workflow_file,
                input_ROI_table="FOV_ROI_table",
                overwrite=False,
            )

    # OME-NGFF JSON validation
    image_zarr = zarr_path / metadata["image"][0]
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
    validate_axes_and_coordinateTransformations(image_zarr)
    validate_axes_and_coordinateTransformations(label_zarr)

    # Load measurements
    meas = ad.read_zarr(
        str(zarr_path / metadata["image"][0] / "tables/regionprops_DAPI/")
    )
    debug(meas.var_names)
    assert "area" in meas.var_names
    assert "bbox_area" in meas.var_names


def test_napari_worfklow_label_input_only(
    tmp_path: Path,
    testdata_path: Path,
    zenodo_zarr: list[str],
    zenodo_zarr_metadata: list[dict[str, Any]],
):

    # Prepare 3D zarr
    zarr_path = tmp_path / "tmp_out/"
    metadata = prepare_3D_zarr(
        str(zarr_path), zenodo_zarr, zenodo_zarr_metadata
    )
    debug(zarr_path)
    debug(metadata)

    # First napari-workflows task (labeling)
    workflow_file = str(testdata_path / "napari_workflows/wf_1.yaml")
    input_specs: dict[str, NapariWorkflowsInput] = {
        "input": {"type": "image", "channel": {"wavelength_id": "A01_C01"}},  # type: ignore # noqa
    }
    output_specs: dict[str, NapariWorkflowsOutput] = {
        "Result of Expand labels (scikit-image, nsbatwm)": {  # type: ignore # noqa
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
            input_ROI_table="FOV_ROI_table",
            level=2,
        )
    debug(metadata)

    # Second napari-workflows task (measurement)
    workflow_file = str(
        testdata_path / "napari_workflows" / "wf_from_labels_to_labels.yaml"
    )
    input_specs = {
        "test_labels": {"type": "label", "label_name": "label_DAPI"},  # type: ignore # noqa
    }
    output_specs = {
        "Result of Expand labels (scikit-image, nsbatwm)": {  # type: ignore # noqa
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
            input_ROI_table="FOV_ROI_table",
        )
    debug(metadata)

    # OME-NGFF JSON validation
    image_zarr = zarr_path / metadata["image"][0]
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
input_specs = dict(
    input_image={"type": "image", "channel": {"wavelength_id": "A01_C01"}}
)
output_specs = dict(output_label={"type": "label", "label_name": LABEL_NAME})
RELABELING_CASE_1: list = [workflow_file_name, input_specs, output_specs]
# 2. Measurement-only workflow, from images+labels to dataframes.
workflow_file_name = "wf_relab_2-measurement_only.yaml"
input_specs = dict(
    input_image={"type": "image", "channel": {"wavelength_id": "A01_C01"}},
    input_label={"type": "label", "label_name": LABEL_NAME},
)
output_specs = dict(
    output_dataframe={"type": "dataframe", "table_name": TABLE_NAME}
)
RELABELING_CASE_2: list = [workflow_file_name, input_specs, output_specs]
# 3. Mixed labeling/measurement workflow.
workflow_file_name = "wf_relab_3-labeling_and_measurement.yaml"
input_specs = dict(
    input_image={"type": "image", "channel": {"wavelength_id": "A01_C01"}}
)
output_specs = dict(
    output_label={"type": "label", "label_name": LABEL_NAME},
    output_dataframe={"type": "dataframe", "table_name": TABLE_NAME},
)
RELABELING_CASE_3: list = [workflow_file_name, input_specs, output_specs]
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
    input_specs: dict[str, dict],
    output_specs: dict[str, dict],
    needs_labels: bool,
    tmp_path: Path,
    testdata_path: Path,
    zenodo_zarr: list[str],
    zenodo_zarr_metadata: list[dict[str, Any]],
):

    # Prepare 3D zarr
    zarr_path = tmp_path / "tmp_out/"
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
                input_ROI_table="FOV_ROI_table",
            )

    # Run napari-workflow for the first time
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
            input_ROI_table="FOV_ROI_table",
        )
    debug(metadata)

    # Check output
    image_zarr = Path(zarr_path / metadata["image"][0])
    validate_labels_and_measurements(
        image_zarr, label_name=LABEL_NAME, table_name=TABLE_NAME
    )

    dataframe_outputs = [
        item for item in output_specs.values() if item["type"] == "dataframe"
    ]
    if dataframe_outputs:
        meas = ad.read_zarr(
            zarr_path / metadata["image"][0] / f"tables/{TABLE_NAME}/"
        )
        debug(meas.var_names)
        assert "area" in meas.var_names
        assert "bbox_area" in meas.var_names


def test_fail_if_no_relabeling(
    tmp_path: Path,
    testdata_path: Path,
    zenodo_zarr: list[str],
    zenodo_zarr_metadata: list[dict[str, Any]],
):

    # Prepare 3D zarr
    zarr_path = tmp_path / "tmp_out/"
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
            input_ROI_table="FOV_ROI_table",
            relabeling=False,
        )
    debug(metadata)

    image_zarr = zarr_path / metadata["image"][0]
    with pytest.raises(AssertionError):
        validate_labels_and_measurements(
            image_zarr, label_name=LABEL_NAME, table_name=TABLE_NAME
        )


cases = [
    (2, 2, True, True),
    (2, 2, False, True),
    (3, 2, True, False),
    (3, 2, False, True),
    (2, 3, False, False),
    (3, 3, False, True),
]


@pytest.mark.parametrize(
    "expected_dimensions,zarr_dimensions,make_CYX,expected_success", cases
)
def test_expected_dimensions(
    expected_dimensions: int,
    zarr_dimensions: int,
    make_CYX: bool,
    expected_success: bool,
    tmp_path: Path,
    testdata_path: Path,
    zenodo_zarr: list[str],
    zenodo_zarr_metadata: list[dict[str, Any]],
):

    # Prepare zarr
    zarr_path = tmp_path / "tmp_out/"
    if zarr_dimensions == 2:
        metadata = prepare_2D_zarr(
            str(zarr_path),
            zenodo_zarr,
            zenodo_zarr_metadata,
            remove_labels=True,
            make_CYX=make_CYX,
        )
    else:
        if make_CYX:
            raise ValueError(f"{make_CYX=} and {zarr_dimensions=}")
        metadata = prepare_3D_zarr(
            str(zarr_path), zenodo_zarr, zenodo_zarr_metadata
        )
    debug(zarr_path)
    debug(metadata)

    # First napari-workflows task (labeling)
    workflow_file = str(
        testdata_path / "napari_workflows/wf_5-labeling_only.yaml"
    )
    input_specs: dict[str, NapariWorkflowsInput] = {
        "input_image": {  # type: ignore # noqa
            "type": "image",
            "channel": {"wavelength_id": "A01_C01"},
        },
    }
    output_specs: dict[str, NapariWorkflowsOutput] = {
        "output_label": {  # type: ignore # noqa
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
            input_ROI_table="FOV_ROI_table",
            level=3,
            expected_dimensions=expected_dimensions,
        )
        if expected_success:
            napari_workflows_wrapper(**arguments)
        else:
            with pytest.raises(ValueError) as e:
                napari_workflows_wrapper(**arguments)
            debug(e.value)


def test_napari_workflow_empty_input_ROI_table(
    tmp_path: Path,
    testdata_path: Path,
    zenodo_zarr: list[str],
    zenodo_zarr_metadata: list[dict[str, Any]],
):
    """
    Run the napari_workflows task, iterating over an empty table of ROIs
    """

    # Init
    zarr_path = tmp_path / "tmp_out/"
    metadata = prepare_3D_zarr(
        str(zarr_path), zenodo_zarr, zenodo_zarr_metadata
    )
    debug(zarr_path)
    debug(metadata)

    # Prepare empty ROI table
    TABLE_NAME = "empty_ROI_table"
    _add_empty_ROI_table(
        image_zarr_path=Path(zarr_path / metadata["image"][0]),
        table_name=TABLE_NAME,
    )

    # First napari-workflows task (labeling)
    workflow_file = str(testdata_path / "napari_workflows/wf_1.yaml")
    input_specs: dict[str, NapariWorkflowsInput] = {
        "input": {"type": "image", "channel": {"wavelength_id": "A01_C01"}},  # type: ignore # noqa
    }
    output_specs: dict[str, NapariWorkflowsOutput] = {
        "Result of Expand labels (scikit-image, nsbatwm)": {  # type: ignore # noqa
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
            input_ROI_table=TABLE_NAME,
            input_specs=input_specs,
            output_specs=output_specs,
            workflow_file=workflow_file,
            level=2,
        )
    debug(metadata)

    # Second napari-workflows task (measurement)
    workflow_file = str(testdata_path / "napari_workflows/wf_4.yaml")
    input_specs = {
        "dapi_img": {"type": "image", "channel": {"wavelength_id": "A01_C01"}},  # type: ignore # noqa
        "dapi_label_img": {"type": "label", "label_name": "label_DAPI"},  # type: ignore # noqa
    }
    output_specs = {
        "regionprops_DAPI": {  # type: ignore # noqa
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
            input_ROI_table=TABLE_NAME,
            input_specs=input_specs,
            output_specs=output_specs,
            workflow_file=workflow_file,
        )
    debug(metadata)

    # OME-NGFF JSON validation
    image_zarr = zarr_path / metadata["image"][0]
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
        str(zarr_path / metadata["image"][0] / "tables/regionprops_DAPI/")
    )
    debug(meas.var_names)


def test_napari_workflow_CYX(
    tmp_path: Path,
    testdata_path: Path,
    zenodo_zarr: list[str],
    zenodo_zarr_metadata: list[dict[str, Any]],
):

    # Init
    zarr_path = tmp_path / "tmp_out/"
    metadata = prepare_2D_zarr(
        str(zarr_path),
        zenodo_zarr,
        zenodo_zarr_metadata,
        remove_labels=True,
        make_CYX=True,
    )
    debug(zarr_path)
    debug(metadata)

    # First napari-workflows task (labeling)
    workflow_file = str(testdata_path / "napari_workflows/wf_1.yaml")
    input_specs: dict[str, NapariWorkflowsInput] = {
        "input": {"type": "image", "channel": {"wavelength_id": "A01_C01"}},  # type: ignore # noqa
    }
    output_specs: dict[str, NapariWorkflowsOutput] = {
        "Result of Expand labels (scikit-image, nsbatwm)": {  # type: ignore # noqa
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
            input_ROI_table="FOV_ROI_table",
            expected_dimensions=2,
            level=2,
        )
    debug(metadata)

    # Second napari-workflows task (measurement)
    workflow_file = str(testdata_path / "napari_workflows/wf_4.yaml")
    input_specs = {
        "dapi_img": {"type": "image", "channel": {"wavelength_id": "A01_C01"}},  # type: ignore # noqa
        "dapi_label_img": {"type": "label", "label_name": "label_DAPI"},  # type: ignore # noqa
    }
    output_specs = {
        "regionprops_DAPI": {  # type: ignore # noqa
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
            input_ROI_table="FOV_ROI_table",
            expected_dimensions=2,
        )
    debug(metadata)

    # OME-NGFF JSON validation
    image_zarr = zarr_path / metadata["image"][0]
    well_zarr = image_zarr.parent
    plate_zarr = image_zarr.parents[2]
    label_zarr = image_zarr / "labels/label_DAPI"
    validate_schema(path=str(image_zarr), type="image")
    validate_schema(path=str(well_zarr), type="well")
    validate_schema(path=str(plate_zarr), type="plate")
    validate_schema(path=str(label_zarr), type="label")

    check_file_number(zarr_path=image_zarr, num_axes=3)

    validate_labels_and_measurements(
        image_zarr, label_name="label_DAPI", table_name="regionprops_DAPI"
    )
    validate_axes_and_coordinateTransformations(image_zarr)
    validate_axes_and_coordinateTransformations(label_zarr)

    # Load measurements
    meas = ad.read_zarr(
        str(zarr_path / metadata["image"][0] / "tables/regionprops_DAPI/")
    )
    debug(meas.var_names)
    assert "area" in meas.var_names
    assert "bbox_area" in meas.var_names


def test_napari_workflow_CYX_wrong_dimensions(
    tmp_path: Path,
    testdata_path: Path,
    zenodo_zarr: list[str],
    zenodo_zarr_metadata: list[dict[str, Any]],
):
    """
    This will fail because of wrong expected_dimensions
    """

    # Init
    zarr_path = tmp_path / "tmp_out/"
    metadata = prepare_2D_zarr(
        str(zarr_path),
        zenodo_zarr,
        zenodo_zarr_metadata,
        remove_labels=True,
        make_CYX=True,
    )
    debug(zarr_path)
    debug(metadata)

    # First napari-workflows task (labeling)
    workflow_file = str(testdata_path / "napari_workflows/wf_1.yaml")
    input_specs: dict[str, NapariWorkflowsInput] = {
        "input": {"type": "image", "channel": {"wavelength_id": "A01_C01"}},  # type: ignore # noqa
    }
    output_specs: dict[str, NapariWorkflowsOutput] = {
        "Result of Expand labels (scikit-image, nsbatwm)": {  # type: ignore # noqa
            "type": "label",
            "label_name": "label_DAPI",
        },
    }
    for component in metadata["image"]:
        with pytest.raises(ValueError) as e:
            napari_workflows_wrapper(
                input_paths=[str(zarr_path)],
                output_path=str(zarr_path),
                metadata=metadata,
                component=component,
                input_specs=input_specs,
                output_specs=output_specs,
                workflow_file=workflow_file,
                input_ROI_table="FOV_ROI_table",
                expected_dimensions=3,
                level=2,
            )
        debug(e.value)


@pytest.mark.skipif(
    not has_napari_skimage_regionprops_mock,
    reason="napari_skimage_regionprops_mock not available",
)
def test_napari_workflow_mock(
    tmp_path: Path,
    testdata_path: Path,
    zenodo_zarr: list[str],
    zenodo_zarr_metadata: list[dict[str, Any]],
):

    # Init
    zarr_path = tmp_path / "tmp_out/"
    metadata = prepare_3D_zarr(
        str(zarr_path), zenodo_zarr, zenodo_zarr_metadata
    )
    debug(zarr_path)
    debug(metadata)

    # Prepare parameters for first napari-workflows task (labeling)
    workflow_file = str(testdata_path / "napari_workflows/wf_1.yaml")
    input_specs: dict[str, NapariWorkflowsInput] = {
        "input": {"type": "image", "channel": {"wavelength_id": "A01_C01"}},  # type: ignore # noqa
    }
    output_specs: dict[str, NapariWorkflowsOutput] = {
        "Result of Expand labels (scikit-image, nsbatwm)": {  # type: ignore # noqa
            "type": "label",
            "label_name": "label_DAPI",
        },
    }

    # Run once
    for component in metadata["image"]:
        napari_workflows_wrapper(
            input_paths=[str(zarr_path)],
            output_path=str(zarr_path),
            metadata=metadata,
            component=component,
            input_specs=input_specs,
            output_specs=output_specs,
            workflow_file=workflow_file,
            input_ROI_table="FOV_ROI_table",
            level=2,
        )
    debug(metadata)

    # Prepare parameters for second napari-workflows task (measurement)
    workflow_file = str(testdata_path / "napari_workflows/wf_4_mock.yaml")
    input_specs = {
        "dapi_img": {"type": "image", "channel": {"wavelength_id": "A01_C01"}},  # type: ignore # noqa
        "dapi_label_img": {"type": "label", "label_name": "label_DAPI"},  # type: ignore # noqa
    }
    output_specs = {
        "regionprops_DAPI": {  # type: ignore # noqa
            "type": "dataframe",
            "table_name": "regionprops_DAPI",
        },
    }

    # Run once
    for component in metadata["image"]:
        napari_workflows_wrapper(
            input_paths=[str(zarr_path)],
            output_path=str(zarr_path),
            metadata=metadata,
            component=component,
            input_specs=input_specs,
            output_specs=output_specs,
            workflow_file=workflow_file,
            input_ROI_table="FOV_ROI_table",
        )
    debug(metadata)

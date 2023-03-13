import logging

import pytest

from fractal_tasks_core.napari_workflows_wrapper import (
    napari_workflows_wrapper,
)


def test_input_specs(tmp_path, testdata_path):
    """
    WHEN calling napari_workflows_wrapper with invalid input_specs
    THEN raise ValueError
    """

    # napari-workflows
    workflow_file = str(
        testdata_path / "napari_workflows/wf_5-labeling_only.yaml"
    )
    input_specs = {"asd": "asd"}
    output_specs = {
        "output_label": {"type": "label", "label_name": "label_DAPI"}
    }
    with pytest.raises(ValueError):
        napari_workflows_wrapper(
            input_paths=[tmp_path],
            output_path=tmp_path,
            metadata={},
            component="component",
            input_specs=input_specs,
            output_specs=output_specs,
            workflow_file=workflow_file,
            input_ROI_table="FOV_ROI_table",
        )


def test_output_specs(tmp_path, testdata_path, caplog):
    """
    WHEN calling napari_workflows_wrapper with invalid output_specs
    THEN raise a Warning
    """
    caplog.set_level(logging.WARNING)

    # napari-workflows
    workflow_file = str(
        testdata_path / "napari_workflows/wf_5-labeling_only.yaml"
    )
    input_specs = {
        "input_image": {"type": "image", "wavelength_id": "A01_C01"}
    }
    output_specs = {"asd": "asd"}

    try:
        napari_workflows_wrapper(
            input_paths=[tmp_path],
            output_path=tmp_path,
            metadata={},
            component="component",
            input_specs=input_specs,
            output_specs=output_specs,
            workflow_file=workflow_file,
            input_ROI_table="FOV_ROI_table",
        )
    except Exception:
        # The task will now fail for some other reason (its arguments are not
        # valid), but we only care about the warning
        pass

    assert "WARNING" in caplog.text
    assert "Some item of wf.leafs" in caplog.text
    assert "is not part of output_specs" in caplog.text


def test_level_setting_in_non_labeling_worfklow(tmp_path, testdata_path):
    """
    WHEN calling napari_workflows_wrapper with a non-labeling-only workflow
         and level>0
    THEN raise NotImplementedError
    """

    # napari-workflows
    workflow_file = str(testdata_path / "napari_workflows/wf_3.yaml")
    input_specs = {
        "slice_img": {"type": "image", "wavelength_id": "A01_C01"},
        "slice_img_c2": {"type": "image", "wavelength_id": "A01_C01"},
    }
    output_specs = {
        "Result of Expand labels (scikit-image, nsbatwm)": {
            "type": "label",
            "label_name": "label_DAPI",
        },
        "regionprops_DAPI": {"type": "dataframe", "table_name": "test"},
    }

    with pytest.raises(NotImplementedError):
        napari_workflows_wrapper(
            input_paths=[tmp_path],
            output_path=tmp_path,
            metadata={},
            component="component",
            input_specs=input_specs,
            output_specs=output_specs,
            workflow_file=workflow_file,
            input_ROI_table="FOV_ROI_table",
            level=2,
        )

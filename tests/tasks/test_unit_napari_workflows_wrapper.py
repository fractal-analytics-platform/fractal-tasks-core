import logging

import pytest
from devtools import debug
from pydantic.v1.error_wrappers import ValidationError

from fractal_tasks_core.tasks.napari_workflows_wrapper import (
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
    zarr_url = str(tmp_path / "component")
    with pytest.raises(ValidationError):
        napari_workflows_wrapper(
            zarr_url=zarr_url,
            input_specs=input_specs,
            output_specs=output_specs,
            workflow_file=workflow_file,
            input_ROI_table="FOV_ROI_table",
        )


def test_output_specs(tmp_path, testdata_path, caplog):
    """
    WHEN
        calling napari_workflows_wrapper with a mismatch between wf.leafs and
        output_specs
    THEN raise a Warning
    """
    caplog.set_level(logging.WARNING)

    # napari-workflows
    workflow_file = str(
        testdata_path / "napari_workflows/wf_5-labeling_only.yaml"
    )
    input_specs = {
        "input_image": {
            "type": "image",
            "channel": {"wavelength_id": "A01_C01"},
        }
    }
    output_specs = {"some_output": {"type": "label", "label_name": "xxx"}}
    zarr_url = str(tmp_path / "component")

    try:
        napari_workflows_wrapper(
            zarr_url=zarr_url,
            input_specs=input_specs,
            output_specs=output_specs,
            workflow_file=workflow_file,
            input_ROI_table="FOV_ROI_table",
        )
    except Exception as e:
        # The task will fail, but we only care about the warning
        debug(e)

    debug(caplog.text)

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
        "slice_img": {
            "type": "image",
            "channel": {"wavelength_id": "A01_C01"},
        },
        "slice_img_c2": {
            "type": "image",
            "channel": {"wavelength_id": "A01_C01"},
        },
    }
    output_specs = {
        "Result of Expand labels (scikit-image, nsbatwm)": {
            "type": "label",
            "label_name": "label_DAPI",
        },
        "regionprops_DAPI": {
            "type": "dataframe",
            "table_name": "test",
            "label_name": "label_DAPI",
        },
    }
    zarr_url = str(tmp_path / "component")
    with pytest.raises(NotImplementedError):
        napari_workflows_wrapper(
            zarr_url=zarr_url,
            input_specs=input_specs,
            output_specs=output_specs,
            workflow_file=workflow_file,
            input_ROI_table="FOV_ROI_table",
            level=2,
        )

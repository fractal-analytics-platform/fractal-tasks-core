import pytest

from fractal_tasks_core.napari_workflows_wrapper import (
    napari_workflows_wrapper,
)


def test_input_specs(tmp_path, testdata_path):
    """
    WHEN calling napari_workflows_wrapper with invalid input_specs or
         output_specs
    THEN raise ValueError
    """

    # napari-workflows
    workflow_file = str(testdata_path / "napari_workflows/wf_3.yaml")
    print(workflow_file)
    input_specs = {
        "input": {"type": "image", "channel": "A01_C01"},
    }
    output_specs = {
        "Result of Expand labels (scikit-image, nsbatwm)": {
            "type": "label",
            "label_name": "label_DAPI",
        },
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
            ROI_table_name="FOV_ROI_table",
        )


def test_level_setting_in_non_labeling_worfklow(tmp_path, testdata_path):
    """
    WHEN calling napari_workflows_wrapper with a non-labeling-only workflow
         and level>0
    THEN raise NotImplementedError
    """

    # napari-workflows
    workflow_file = str(testdata_path / "napari_workflows/wf_3.yaml")
    input_specs = {
        "slice_img": {"type": "image", "channel": "A01_C01"},
        "slice_img_c2": {"type": "image", "channel": "A01_C01"},
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
            ROI_table_name="FOV_ROI_table",
            level=2,
        )

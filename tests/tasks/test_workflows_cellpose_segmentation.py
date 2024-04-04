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
import json
import logging
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pytest
import zarr
from devtools import debug
from pytest import MonkeyPatch

import fractal_tasks_core.tasks
from .._zenodo_ome_zarrs import prepare_2D_zarr
from .._zenodo_ome_zarrs import prepare_3D_zarr
from ._validation import check_file_number
from ._validation import validate_axes_and_coordinateTransformations
from ._validation import validate_schema
from .lib_empty_ROI_table import _add_empty_ROI_table
from fractal_tasks_core.channels import ChannelInputModel
from fractal_tasks_core.tasks.cellpose_segmentation import (
    cellpose_segmentation,
)
from fractal_tasks_core.tasks.cellvoyager_to_ome_zarr_compute import (
    cellvoyager_to_ome_zarr_compute,
)
from fractal_tasks_core.tasks.cellvoyager_to_ome_zarr_init import (
    cellvoyager_to_ome_zarr_init,
)
from fractal_tasks_core.tasks.copy_ome_zarr_hcs_plate import (
    copy_ome_zarr_hcs_plate,
)
from fractal_tasks_core.tasks.maximum_intensity_projection import (
    maximum_intensity_projection,
)
from fractal_tasks_core.zarr_utils import OverwriteNotAllowedError

allowed_channels = [
    {
        "label": "DAPI",
        "wavelength_id": "A01_C01",
        "color": "00FFFF",
        "window": {"start": 0, "end": 700},
    },
    {
        "wavelength_id": "A01_C02",
        "label": "nanog",
        "color": "FF00FF",
        "window": {"start": 0, "end": 180},
    },
    {
        "wavelength_id": "A02_C03",
        "label": "Lamin B1",
        "color": "FFFF00",
        "window": {"start": 0, "end": 1500},
    },
]


num_levels = 6
coarsening_xy = 2


def patched_segment_ROI(
    x, do_3D=True, label_dtype=None, well_id=None, **kwargs
):
    # Expects x to always be a 4D image

    import logging

    logger = logging.getLogger("cellpose_segmentation.py")

    logger.info(f"[{well_id}][patched_segment_ROI] START")
    assert x.ndim == 4
    # Actual labeling: segment_ROI returns a 3D mask with the same shape as x,
    # except for the first dimension
    mask = np.zeros_like(x[0, :, :, :])
    nz, ny, nx = mask.shape
    if do_3D:
        mask[:, 0 : ny // 4, 0 : nx // 4] = 1  # noqa
        mask[:, ny // 4 : ny // 2, 0 : int(nx * 0.9)] = 2  # noqa
    else:
        mask[:, 0 : ny // 4, 0 : nx // 4] = 1  # noqa
        mask[:, 0 : ny // 2, 0 : nx // 4] = 1  # noqa
        mask[:, 0 : ny // 4, 0 : nx // 2] = 1  # noqa
        mask[:, int(ny * 3 / 4) : ny, int(nx * 3 / 4) : nx] = 2  # noqa

    logger.info(f"[{well_id}][patched_segment_ROI] END")

    return mask.astype(label_dtype)


def patched_segment_ROI_no_labels(
    x, do_3D=True, label_dtype=None, well_id=None, **kwargs
):
    import logging

    logger = logging.getLogger("cellpose_segmentation.py")
    logger.info(f"[{well_id}][patched_segment_ROI_no_labels] START")
    assert x.ndim == 4
    mask = np.zeros_like(x[0, :, :, :])
    logger.info(f"[{well_id}][patched_segment_ROI_no_labels] END")
    return mask.astype(label_dtype)


def patched_segment_ROI_overlapping_organoids(
    x, label_dtype=None, well_id=None, **kwargs
):

    import logging

    logger = logging.getLogger("cellpose_segmentation.py")
    logger.info(f"[{well_id}][patched_segment_ROI] START")

    assert x.ndim == 4
    # Actual labeling: segment_ROI returns a 3D mask with the same shape as x,
    # except for the first dimension
    mask = np.zeros_like(x[0, :, :, :])
    nz, ny, nx = mask.shape
    indices = np.arange(0, nx // 2)
    mask[:, indices, indices] = 1  # noqa
    mask[:, indices + 10, indices + 20] = 2  # noqa

    logger.info(f"[{well_id}][patched_segment_ROI] END")

    return mask.astype(label_dtype)


def patched_cellpose_core_use_gpu(*args, **kwargs):
    debug("WARNING: using patched_cellpose_core_use_gpu")
    return False


def test_failures(
    tmp_path: Path,
    testdata_path: Path,
    zenodo_zarr: list[str],
    zenodo_zarr_metadata: list[dict[str, Any]],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: MonkeyPatch,
):

    monkeypatch.setattr(
        "fractal_tasks_core.tasks.cellpose_segmentation.cellpose.core.use_gpu",
        patched_cellpose_core_use_gpu,
    )

    monkeypatch.setattr(
        "fractal_tasks_core.tasks.cellpose_segmentation.segment_ROI",
        patched_segment_ROI,
    )

    caplog.set_level(logging.WARNING)

    # Use pre-made 3D zarr
    zarr_dir = tmp_path / "tmp_out/"
    metadata = prepare_3D_zarr(
        str(zarr_dir), zenodo_zarr, zenodo_zarr_metadata
    )
    debug(zarr_dir)
    debug(metadata)

    # A sequence of invalid attempts
    for component in metadata["image"]:
        zarr_url = str(zarr_dir / component)
        kwargs = dict(
            zarr_url=zarr_url,
            level=3,
        )
        # Attempt 1
        cellpose_segmentation(
            **kwargs,
            channel=ChannelInputModel(wavelength_id="invalid_wavelength_id"),
        )
        assert "ChannelNotFoundError" in caplog.records[0].msg

        # Attempt 2
        cellpose_segmentation(
            **kwargs,
            channel=ChannelInputModel(label="invalid_channel_name"),
        )
        assert "ChannelNotFoundError" in caplog.records[0].msg
        assert "ChannelNotFoundError" in caplog.records[1].msg

        # Attempt 3
        with pytest.raises(ValueError):
            cellpose_segmentation(
                **kwargs,
                channel=ChannelInputModel(
                    wavelength_id="A01_C01",
                    label="invalid_channel_name",
                ),
            )


def test_workflow_with_per_FOV_labeling(
    tmp_path: Path,
    testdata_path: Path,
    zenodo_zarr: list[str],
    zenodo_zarr_metadata: list[dict[str, Any]],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: MonkeyPatch,
):

    monkeypatch.setattr(
        "fractal_tasks_core.tasks.cellpose_segmentation.cellpose.core.use_gpu",
        patched_cellpose_core_use_gpu,
    )

    monkeypatch.setattr(
        "fractal_tasks_core.tasks.cellpose_segmentation.segment_ROI",
        patched_segment_ROI,
    )

    # Setup caplog fixture, see
    # https://docs.pytest.org/en/stable/how-to/logging.html#caplog-fixture
    caplog.set_level(logging.INFO)

    # Use pre-made 3D zarr
    zarr_dir = tmp_path / "tmp_out/"
    metadata = prepare_3D_zarr(
        str(zarr_dir), zenodo_zarr, zenodo_zarr_metadata
    )
    debug(zarr_dir)
    debug(metadata)

    # Per-FOV labeling
    for component in metadata["image"]:
        zarr_url = str(zarr_dir / component)
        cellpose_segmentation(
            zarr_url=zarr_url,
            channel=ChannelInputModel(wavelength_id="A01_C01"),
            level=3,
            relabeling=True,
            diameter_level0=80.0,
            augment=True,
            net_avg=True,
            min_size=30,
        )

    # OME-NGFF JSON validation
    image_zarr = Path(zarr_dir / metadata["image"][0])
    label_zarr = image_zarr / "labels/label_DAPI"
    well_zarr = image_zarr.parent
    plate_zarr = image_zarr.parents[2]
    validate_schema(path=str(image_zarr), type="image")
    validate_schema(path=str(well_zarr), type="well")
    validate_schema(path=str(plate_zarr), type="plate")
    validate_schema(path=str(label_zarr), type="label")

    check_file_number(zarr_path=image_zarr)


def test_workflow_with_multi_channel_input(
    tmp_path: Path,
    testdata_path: Path,
    zenodo_zarr: list[Path],
    zenodo_zarr_metadata: list[dict[str, Any]],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: MonkeyPatch,
):
    # Testing by providing the same channel twice as wavelength_id &
    # wavelength_id_c2

    monkeypatch.setattr(
        "fractal_tasks_core.tasks.cellpose_segmentation.cellpose.core.use_gpu",
        patched_cellpose_core_use_gpu,
    )

    monkeypatch.setattr(
        "fractal_tasks_core.tasks.cellpose_segmentation.segment_ROI",
        patched_segment_ROI,
    )

    # Setup caplog fixture, see
    # https://docs.pytest.org/en/stable/how-to/logging.html#caplog-fixture
    caplog.set_level(logging.INFO)

    # Use pre-made 3D zarr
    zarr_dir = tmp_path / "tmp_out/"
    metadata = prepare_3D_zarr(
        str(zarr_dir), [str(x) for x in zenodo_zarr], zenodo_zarr_metadata
    )
    debug(zarr_dir)
    debug(metadata)

    # Per-FOV labeling
    for component in metadata["image"]:
        zarr_url = str(zarr_dir / component)
        cellpose_segmentation(
            zarr_url=zarr_url,
            channel=ChannelInputModel(wavelength_id="A01_C01"),
            channel2=ChannelInputModel(wavelength_id="A01_C01"),
            level=3,
            relabeling=True,
            diameter_level0=80.0,
            model_type="cyto2",
        )

    # OME-NGFF JSON validation
    image_zarr = Path(zarr_dir / metadata["image"][0])
    label_zarr = image_zarr / "labels/label_DAPI"
    well_zarr = image_zarr.parent
    plate_zarr = image_zarr.parents[2]
    validate_schema(path=str(image_zarr), type="image")
    validate_schema(path=str(well_zarr), type="well")
    validate_schema(path=str(plate_zarr), type="plate")
    validate_schema(path=str(label_zarr), type="label")

    check_file_number(zarr_path=image_zarr)


def test_workflow_with_per_FOV_labeling_2D(
    tmp_path: Path,
    testdata_path: Path,
    zenodo_zarr: list[str],
    zenodo_zarr_metadata: list[dict[str, Any]],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: MonkeyPatch,
):

    monkeypatch.setattr(
        "fractal_tasks_core.tasks.cellpose_segmentation.cellpose.core.use_gpu",
        patched_cellpose_core_use_gpu,
    )

    # Do not use cellpose
    monkeypatch.setattr(
        "fractal_tasks_core.tasks.cellpose_segmentation.segment_ROI",
        patched_segment_ROI,
    )

    # Load pre-made 2D zarr array
    zarr_dir_mip = tmp_path / "tmp_out_mip/"
    metadata = prepare_2D_zarr(
        str(zarr_dir_mip),
        zenodo_zarr,
        zenodo_zarr_metadata,
        remove_labels=True,
    )

    # Per-FOV labeling
    for component in metadata["image"]:
        zarr_url = str(zarr_dir_mip / component)
        cellpose_segmentation(
            zarr_url=zarr_url,
            channel=ChannelInputModel(wavelength_id="A01_C01"),
            level=2,
            relabeling=True,
            diameter_level0=80.0,
        )

    # OME-NGFF JSON validation
    image_zarr = Path(zarr_dir_mip / metadata["image"][0])
    debug(image_zarr)
    well_zarr = image_zarr.parent
    plate_zarr = image_zarr.parents[2]
    validate_schema(path=str(image_zarr), type="image")
    validate_schema(path=str(well_zarr), type="well")
    validate_schema(path=str(plate_zarr), type="plate")

    check_file_number(zarr_path=image_zarr)


def test_workflow_with_per_well_labeling_2D(
    tmp_path: Path,
    testdata_path: Path,
    zenodo_images: str,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: MonkeyPatch,
):

    monkeypatch.setattr(
        "fractal_tasks_core.tasks.cellpose_segmentation.cellpose.core.use_gpu",
        patched_cellpose_core_use_gpu,
    )

    # Do not use cellpose
    monkeypatch.setattr(
        "fractal_tasks_core.tasks.cellpose_segmentation.segment_ROI",
        patched_segment_ROI,
    )

    # Init
    img_path = Path(zenodo_images)
    zarr_dir = tmp_path / "tmp_out/"

    # Create zarr structure
    parallelization_list = cellvoyager_to_ome_zarr_init(
        zarr_urls=[],
        zarr_dir=str(zarr_dir),
        image_dirs=[str(img_path)],
        image_extension="png",
        allowed_channels=allowed_channels,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        metadata_table_file=None,
    )

    # Yokogawa to zarr
    image_list_updates = []
    for image in parallelization_list:
        image_list_updates += cellvoyager_to_ome_zarr_compute(
            zarr_url=image["zarr_url"],
            init_args=image["init_args"],
        )["image_list_updates"]
    debug(image_list_updates)

    zarr_urls = []
    for image in image_list_updates:
        zarr_urls.append(image["zarr_url"])

    # Replicate
    parallelization_list = copy_ome_zarr_hcs_plate(
        zarr_urls=zarr_urls,
        zarr_dir=str(zarr_dir),
    )
    debug(parallelization_list)

    # MIP
    image_list_updates = []
    for image in parallelization_list:
        image_list_updates += maximum_intensity_projection(
            zarr_url=image["zarr_url"],
            init_args=image["init_args"],
            overwrite=True,
        )["image_list_updates"]

    # Whole-well labeling
    for zarr_url in zarr_urls:
        cellpose_segmentation(
            zarr_url=zarr_url,
            channel=ChannelInputModel(wavelength_id="A01_C01"),
            level=2,
            input_ROI_table="well_ROI_table",
            relabeling=True,
            diameter_level0=80.0,
        )

    # OME-NGFF JSON validation
    image_zarr = Path(zarr_urls[0])
    label_zarr = image_zarr / "labels/label_DAPI"
    debug(image_zarr)
    well_zarr = image_zarr.parent
    plate_zarr = image_zarr.parents[2]
    validate_schema(path=str(image_zarr), type="image")
    validate_schema(path=str(well_zarr), type="well")
    validate_schema(path=str(plate_zarr), type="plate")

    validate_axes_and_coordinateTransformations(label_zarr)
    validate_axes_and_coordinateTransformations(image_zarr)
    check_file_number(zarr_path=image_zarr)


def test_workflow_bounding_box(
    tmp_path: Path,
    testdata_path: Path,
    zenodo_zarr: list[str],
    zenodo_zarr_metadata: list[dict[str, Any]],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: MonkeyPatch,
):

    monkeypatch.setattr(
        "fractal_tasks_core.tasks.cellpose_segmentation.cellpose.core.use_gpu",
        patched_cellpose_core_use_gpu,
    )

    monkeypatch.setattr(
        "fractal_tasks_core.tasks.cellpose_segmentation.segment_ROI",
        patched_segment_ROI,
    )
    NUM_LABELS = 4

    # Setup caplog fixture, see
    # https://docs.pytest.org/en/stable/how-to/logging.html#caplog-fixture
    caplog.set_level(logging.INFO)

    # Use pre-made 3D zarr
    zarr_dir = tmp_path / "tmp_out/"
    metadata = prepare_3D_zarr(
        str(zarr_dir), zenodo_zarr, zenodo_zarr_metadata
    )
    debug(zarr_dir)
    debug(metadata)

    # Per-FOV labeling
    for component in metadata["image"]:
        zarr_url = str(zarr_dir / component)
        cellpose_segmentation(
            zarr_url=zarr_url,
            channel=ChannelInputModel(wavelength_id="A01_C01"),
            level=3,
            relabeling=True,
            diameter_level0=80.0,
            output_ROI_table="bbox_table",
        )

    # Re-run with overwrite=True
    for component in metadata["image"]:
        zarr_url = str(zarr_dir / component)
        cellpose_segmentation(
            zarr_url=zarr_url,
            channel=ChannelInputModel(wavelength_id="A01_C01"),
            level=3,
            relabeling=True,
            diameter_level0=80.0,
            output_ROI_table="bbox_table",
            overwrite=True,
        )

    # Re-run with overwrite=False
    with pytest.raises(OverwriteNotAllowedError):
        for component in metadata["image"]:
            zarr_url = str(zarr_dir / component)
            cellpose_segmentation(
                zarr_url=zarr_url,
                channel=ChannelInputModel(wavelength_id="A01_C01"),
                level=3,
                relabeling=True,
                diameter_level0=80.0,
                output_ROI_table="bbox_table",
                overwrite=False,
            )

    bbox_ROIs_table_path = str(
        zarr_dir / metadata["image"][0] / "tables/bbox_table/"
    )
    bbox_ROIs = ad.read_zarr(bbox_ROIs_table_path)
    debug(bbox_ROIs)
    debug(bbox_ROIs.X)
    debug(bbox_ROIs.obs)
    assert bbox_ROIs.obs.shape == (NUM_LABELS, 1)
    assert bbox_ROIs.shape == (NUM_LABELS, 6)
    assert len(bbox_ROIs) > 0
    assert np.max(bbox_ROIs.X) == float(416)

    # Add test for fractal-tasks-core issue #560 (some Zarr attributes missing
    # in the ROI-table group)
    table_group = zarr.open_group(bbox_ROIs_table_path)
    debug(table_group.attrs.asdict())
    assert "encoding-type" in table_group.attrs.asdict().keys()
    assert "encoding-version" in table_group.attrs.asdict().keys()


def test_workflow_bounding_box_with_overlap(
    tmp_path: Path,
    testdata_path: Path,
    zenodo_zarr: list[str],
    zenodo_zarr_metadata: list[dict[str, Any]],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: MonkeyPatch,
):

    monkeypatch.setattr(
        "fractal_tasks_core.tasks.cellpose_segmentation.cellpose.core.use_gpu",
        patched_cellpose_core_use_gpu,
    )

    monkeypatch.setattr(
        "fractal_tasks_core.tasks.cellpose_segmentation.segment_ROI",
        patched_segment_ROI_overlapping_organoids,
    )

    # Setup caplog fixture, see
    # https://docs.pytest.org/en/stable/how-to/logging.html#caplog-fixture
    caplog.set_level(logging.WARNING)

    # Use pre-made 3D zarr
    zarr_dir = tmp_path / "tmp_out/"
    metadata = prepare_3D_zarr(
        str(zarr_dir), zenodo_zarr, zenodo_zarr_metadata
    )
    debug(zarr_dir)
    debug(metadata)

    # Per-FOV labeling
    for component in metadata["image"]:
        zarr_url = str(zarr_dir / component)
        cellpose_segmentation(
            zarr_url=zarr_url,
            channel=ChannelInputModel(wavelength_id="A01_C01"),
            level=3,
            relabeling=True,
            diameter_level0=80.0,
            output_ROI_table="bbox_table",
        )
        debug(caplog.text)
        assert "bounding-box pairs overlap" in caplog.text


def test_workflow_with_per_FOV_labeling_via_script(
    tmp_path: Path,
    testdata_path: Path,
    zenodo_zarr: list[str],
    zenodo_zarr_metadata: list[dict[str, Any]],
):
    # Use pre-made 3D zarr
    zarr_dir = tmp_path / "tmp_out/"
    metadata = prepare_3D_zarr(
        str(zarr_dir), zenodo_zarr, zenodo_zarr_metadata
    )
    debug(zarr_dir)
    debug(metadata)

    python_path = sys.executable
    task_path = (
        Path(fractal_tasks_core.tasks.__file__).parent
        / "cellpose_segmentation.py"
    )
    args_path = tmp_path / "args.json"
    out_path = tmp_path / "out.json"
    command = (
        f"{str(python_path)} {str(task_path)} "
        f"-j {str(args_path)} --metadata-out {str(out_path)}"
    )
    debug(command)

    zarr_url = str(zarr_dir / metadata["image"][0])
    task_args = dict(
        zarr_url=zarr_url,
        channel=dict(wavelength_id="A01_C01"),
        level=4,
        relabeling=True,
        diameter_level0=80.0,
        augment=True,
        net_avg=True,
        min_size=30,
        use_gpu=False,
    )

    run_options = dict(timeout=15, capture_output=True, encoding="utf-8")

    # Valid model_type -> should fail due to timeout
    this_task_args = dict(**task_args, model_type="nuclei")
    with args_path.open("w") as f:
        json.dump(this_task_args, f, indent=2)
    with pytest.raises(subprocess.TimeoutExpired):
        res = subprocess.run(shlex.split(command), **run_options)  # type: ignore  # noqa
        print(res.stdout)
        print(res.stderr)
        # Also check that we are not hitting
        # https://github.com/fractal-analytics-platform/fractal-tasks-core/issues/343
        assert "urllib.error.HTTPError" not in res.stdout
        assert "urllib.error.HTTPError" not in res.stderr

    # Invalid model_type -> should fail with ValueError
    INVALID_MODEL_TYPE = "something_wrong"
    this_task_args = dict(**task_args, model_type=INVALID_MODEL_TYPE)
    with args_path.open("w") as f:
        json.dump(this_task_args, f, indent=2)
    res = subprocess.run(shlex.split(command), **run_options)  # type: ignore
    print(res.stdout)
    print(res.stderr)
    # If this check fails after updating the cellpose version, you'll likely
    # need to update the manifest to include a changed set of available models
    # See https://github.com/fractal-analytics-platform/fractal-tasks-core/issues/401 # noqa E501
    error_msg = (
        "unexpected value; permitted: 'cyto', 'nuclei', "
        "'tissuenet', 'livecell', 'cyto2', 'general', 'CP', 'CPx', "
        "'TN1', 'TN2', 'TN3', 'LC1', 'LC2', 'LC3', 'LC4' "
        f"(type=value_error.const; given={INVALID_MODEL_TYPE}; "
        "permitted=('cyto', 'nuclei', 'tissuenet', 'livecell', "
        "'cyto2', 'general', 'CP', 'CPx', 'TN1', 'TN2', 'TN3', "
        "'LC1', 'LC2', 'LC3', 'LC4'))"
    )
    assert error_msg in res.stderr
    assert "urllib.error.HTTPError" not in res.stdout
    assert "urllib.error.HTTPError" not in res.stderr


def test_workflow_with_per_FOV_labeling_with_empty_FOV_table(
    tmp_path: Path,
    testdata_path: Path,
    zenodo_zarr: list[str],
    zenodo_zarr_metadata: list[dict[str, Any]],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: MonkeyPatch,
):
    """
    Run the cellpose task iterating over an empty table of ROIs
    """

    # Use pre-made 3D zarr
    zarr_dir = tmp_path / "tmp_out/"
    metadata = prepare_3D_zarr(
        str(zarr_dir), zenodo_zarr, zenodo_zarr_metadata
    )
    debug(zarr_dir)
    debug(metadata)

    # Prepare empty ROI table
    TABLE_NAME = "empty_ROI_table"
    _add_empty_ROI_table(
        image_zarr_path=Path(zarr_dir / metadata["image"][0]),
        table_name=TABLE_NAME,
    )

    # Per-FOV labeling
    for component in metadata["image"]:
        zarr_url = str(zarr_dir / component)
        cellpose_segmentation(
            zarr_url=zarr_url,
            input_ROI_table=TABLE_NAME,
            channel=ChannelInputModel(wavelength_id="A01_C01"),
            level=3,
            relabeling=True,
            diameter_level0=80.0,
            augment=True,
            net_avg=True,
            min_size=30,
        )

    # OME-NGFF JSON validation
    image_zarr = Path(zarr_dir / metadata["image"][0])
    label_zarr = image_zarr / "labels/label_DAPI"
    well_zarr = image_zarr.parent
    plate_zarr = image_zarr.parents[2]
    validate_schema(path=str(image_zarr), type="image")
    validate_schema(path=str(well_zarr), type="well")
    validate_schema(path=str(plate_zarr), type="plate")
    validate_schema(path=str(label_zarr), type="label")

    check_file_number(zarr_path=image_zarr)


def test_CYX_input(
    tmp_path: Path,
    testdata_path: Path,
    zenodo_zarr: list[str],
    zenodo_zarr_metadata: list[dict[str, Any]],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: MonkeyPatch,
):
    """
    FIXME This test works for the wrong reason (the fact that C and Z scale
    transformations are both equal to 1).
    """

    monkeypatch.setattr(
        "fractal_tasks_core.tasks.cellpose_segmentation.cellpose.core.use_gpu",
        patched_cellpose_core_use_gpu,
    )

    # Do not use cellpose
    monkeypatch.setattr(
        "fractal_tasks_core.tasks.cellpose_segmentation.segment_ROI",
        patched_segment_ROI,
    )

    # Load pre-made 2D zarr array
    zarr_dir_mip = tmp_path / "tmp_out_mip/"
    metadata = prepare_2D_zarr(
        str(zarr_dir_mip),
        zenodo_zarr,
        zenodo_zarr_metadata,
        remove_labels=True,
        make_CYX=True,
    )

    # Per-FOV labeling
    for component in metadata["image"]:
        zarr_url = str(zarr_dir_mip / component)
        cellpose_segmentation(
            zarr_url=zarr_url,
            channel=ChannelInputModel(wavelength_id="A01_C01"),
            level=0,
            relabeling=True,
            diameter_level0=80.0,
        )

    # OME-NGFF JSON validation
    image_zarr = Path(zarr_dir_mip / metadata["image"][0])
    label_zarr = image_zarr / "labels/label_DAPI"
    debug(image_zarr)
    well_zarr = image_zarr.parent
    plate_zarr = image_zarr.parents[2]
    validate_schema(path=str(image_zarr), type="image")
    validate_schema(path=str(well_zarr), type="well")
    validate_schema(path=str(plate_zarr), type="plate")
    validate_axes_and_coordinateTransformations(image_zarr)
    validate_axes_and_coordinateTransformations(label_zarr)


def test_workflow_secondary_labeling(
    tmp_path: Path,
    testdata_path: Path,
    zenodo_zarr: list[str],
    zenodo_zarr_metadata: list[dict[str, Any]],
    monkeypatch: MonkeyPatch,
):

    monkeypatch.setattr(
        "fractal_tasks_core.tasks.cellpose_segmentation.cellpose.core.use_gpu",
        patched_cellpose_core_use_gpu,
    )

    monkeypatch.setattr(
        "fractal_tasks_core.tasks.cellpose_segmentation.segment_ROI",
        patched_segment_ROI,
    )

    # Load pre-made 2D zarr array
    zarr_dir = tmp_path / "tmp_out_mip/"
    metadata = prepare_2D_zarr(
        str(zarr_dir),
        zenodo_zarr,
        zenodo_zarr_metadata,
        remove_labels=True,
        make_CYX=False,
    )

    # Primary segmentation (organoid)
    for component in metadata["image"]:
        zarr_url = str(zarr_dir / component)
        cellpose_segmentation(
            zarr_url=zarr_url,
            channel=ChannelInputModel(wavelength_id="A01_C01"),
            level=0,
            relabeling=True,
            input_ROI_table="FOV_ROI_table",
            output_label_name="organoids",
            output_ROI_table="organoid_ROI_table",
        )

    organoid_ROI_table_path = str(
        zarr_dir / metadata["image"][0] / "tables/organoid_ROI_table"
    )
    organoid_ROI_table = ad.read_zarr(organoid_ROI_table_path)
    organoid_ROI_table_zarr_group = zarr.open(organoid_ROI_table_path)
    debug(organoid_ROI_table_zarr_group.attrs.asdict())
    debug(organoid_ROI_table)
    debug(organoid_ROI_table.X)
    NUM_LABELS_PER_FOV = 2
    assert organoid_ROI_table.obs.shape == (NUM_LABELS_PER_FOV * 2, 1)
    assert organoid_ROI_table.shape == (NUM_LABELS_PER_FOV * 2, 6)
    assert len(organoid_ROI_table) > 0
    assert np.max(organoid_ROI_table.X) == float(728)

    debug(zarr_dir / metadata["image"][0])

    # Secondary segmentation (nuclei)
    for component in metadata["image"]:
        zarr_url = str(zarr_dir / component)
        cellpose_segmentation(
            zarr_url=zarr_url,
            channel=ChannelInputModel(wavelength_id="A01_C01"),
            level=0,
            relabeling=True,
            input_ROI_table="organoid_ROI_table",
            use_masks=True,
            output_label_name="nuclei",
        )
    # FIXME: what could we assert here?


def test_workflow_secondary_labeling_no_labels(
    tmp_path: Path,
    testdata_path: Path,
    zenodo_zarr: list[str],
    zenodo_zarr_metadata: list[dict[str, Any]],
    monkeypatch: MonkeyPatch,
):
    """
    Run a first Cellpose segmentation that produces no labels, and therefore
    writes an empty ROI table into `organoid_ROI_table`. Then run another
    Cellpose segmentation with `input_ROI_table="organoid_ROI_table"`, which
    loops over an empty list of ROIs and then produces an empty list of
    bounding-box DataFrames.

    This test is to check that running with an empty input ROI table does not
    raise any error (see issue #561 in fractal-tasks-core).
    """

    monkeypatch.setattr(
        "fractal_tasks_core.tasks.cellpose_segmentation.cellpose.core.use_gpu",
        patched_cellpose_core_use_gpu,
    )

    monkeypatch.setattr(
        "fractal_tasks_core.tasks.cellpose_segmentation.segment_ROI",
        patched_segment_ROI_no_labels,
    )

    # Load pre-made 2D zarr array
    zarr_dir = tmp_path / "tmp_out_mip/"
    metadata = prepare_2D_zarr(
        str(zarr_dir),
        zenodo_zarr,
        zenodo_zarr_metadata,
        remove_labels=True,
        make_CYX=False,
    )

    # Primary segmentation (organoid)
    for component in metadata["image"]:
        zarr_url = str(zarr_dir / component)
        cellpose_segmentation(
            zarr_url=zarr_url,
            channel=ChannelInputModel(wavelength_id="A01_C01"),
            level=0,
            relabeling=True,
            input_ROI_table="FOV_ROI_table",
            output_label_name="organoids",
            output_ROI_table="organoid_ROI_table",
        )

    organoid_ROI_table_path = str(
        zarr_dir / metadata["image"][0] / "tables/organoid_ROI_table"
    )
    organoid_ROI_table = ad.read_zarr(organoid_ROI_table_path)
    organoid_ROI_table_zarr_group = zarr.open(organoid_ROI_table_path)
    debug(organoid_ROI_table_zarr_group.attrs.asdict())
    debug(organoid_ROI_table)
    debug(organoid_ROI_table.X)
    debug(organoid_ROI_table.shape)
    assert len(organoid_ROI_table) == 0

    debug(zarr_dir / metadata["image"][0])

    # Secondary segmentation (nuclei)
    for component in metadata["image"]:
        zarr_url = str(zarr_dir / component)
        cellpose_segmentation(
            zarr_url=zarr_url,
            channel=ChannelInputModel(wavelength_id="A01_C01"),
            level=0,
            relabeling=True,
            input_ROI_table="organoid_ROI_table",
            output_ROI_table="nuclei_ROI_table",
            use_masks=True,
            output_label_name="nuclei",
        )


def test_workflow_secondary_labeling_two_channels(
    tmp_path: Path,
    testdata_path: Path,
    zenodo_zarr: list[str],
    zenodo_zarr_metadata: list[dict[str, Any]],
    monkeypatch: MonkeyPatch,
):
    """
    Test that catches issue
    https://github.com/fractal-analytics-platform/fractal-tasks-core/issues/640,
    namely the fact the cellpose_segmentation with masked-loading fails when
    using more than one channel.
    """

    monkeypatch.setattr(
        "fractal_tasks_core.tasks.cellpose_segmentation.cellpose.core.use_gpu",
        patched_cellpose_core_use_gpu,
    )

    monkeypatch.setattr(
        "fractal_tasks_core.tasks.cellpose_segmentation.segment_ROI",
        patched_segment_ROI,
    )

    # Load pre-made 2D zarr array
    zarr_dir = tmp_path / "tmp_out_mip/"
    metadata = prepare_2D_zarr(
        str(zarr_dir),
        zenodo_zarr,
        zenodo_zarr_metadata,
        remove_labels=True,
        make_CYX=False,
    )

    # Primary segmentation (organoid)
    for component in metadata["image"]:
        zarr_url = str(zarr_dir / component)
        cellpose_segmentation(
            zarr_url=zarr_url,
            channel=ChannelInputModel(wavelength_id="A01_C01"),
            level=0,
            relabeling=True,
            input_ROI_table="FOV_ROI_table",
            output_label_name="organoids",
            output_ROI_table="organoid_ROI_table",
        )

    organoid_ROI_table_path = str(
        zarr_dir / metadata["image"][0] / "tables/organoid_ROI_table"
    )
    organoid_ROI_table = ad.read_zarr(organoid_ROI_table_path)
    organoid_ROI_table_zarr_group = zarr.open(organoid_ROI_table_path)
    debug(organoid_ROI_table_zarr_group.attrs.asdict())
    debug(organoid_ROI_table)
    debug(organoid_ROI_table.X)
    NUM_LABELS_PER_FOV = 2
    assert organoid_ROI_table.obs.shape == (NUM_LABELS_PER_FOV * 2, 1)
    assert organoid_ROI_table.shape == (NUM_LABELS_PER_FOV * 2, 6)
    assert len(organoid_ROI_table) > 0
    assert np.max(organoid_ROI_table.X) == float(728)

    debug(zarr_dir / metadata["image"][0])

    # Secondary segmentation (nuclei)
    for component in metadata["image"]:
        zarr_url = str(zarr_dir / component)
        cellpose_segmentation(
            zarr_url=zarr_url,
            channel=ChannelInputModel(wavelength_id="A01_C01"),
            channel2=ChannelInputModel(wavelength_id="A01_C01"),
            level=0,
            relabeling=True,
            input_ROI_table="organoid_ROI_table",
            use_masks=True,
            output_label_name="nuclei",
        )

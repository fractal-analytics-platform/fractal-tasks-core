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
import logging
import shutil
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List

import anndata as ad
import numpy as np
import pytest
from devtools import debug
from pytest import MonkeyPatch

from .utils import check_file_number
from .utils import validate_schema
from fractal_tasks_core.cellpose_segmentation import cellpose_segmentation
from fractal_tasks_core.copy_ome_zarr import (
    copy_ome_zarr,
)  # noqa
from fractal_tasks_core.create_ome_zarr import create_ome_zarr
from fractal_tasks_core.maximum_intensity_projection import (
    maximum_intensity_projection,
)  # noqa
from fractal_tasks_core.yokogawa_to_ome_zarr import yokogawa_to_ome_zarr


allowed_channels = [
    {
        "label": "DAPI",
        "wavelength_id": "A01_C01",
        "colormap": "00FFFF",
        "start": 0,
        "end": 700,
    },
    {
        "wavelength_id": "A01_C02",
        "label": "nanog",
        "colormap": "FF00FF",
        "start": 0,
        "end": 180,
    },
    {
        "wavelength_id": "A02_C03",
        "label": "Lamin B1",
        "colormap": "FFFF00",
        "start": 0,
        "end": 1500,
    },
]


num_levels = 6
coarsening_xy = 2


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


def patched_segment_FOV(
    x, do_3D=True, label_dtype=None, well_id=None, **kwargs
):

    import logging

    logger = logging.getLogger("cellpose_segmentation.py")

    logger.info(f"[{well_id}][patched_segment_FOV] START")

    # Actual labeling
    mask = np.zeros_like(x)
    nz, ny, nx = mask.shape
    if do_3D:
        mask[:, 0 : ny // 4, 0 : nx // 4] = 1  # noqa
        mask[:, ny // 4 : ny // 2, 0 : nx // 2] = 2  # noqa
    else:
        mask[:, 0 : ny // 4, 0 : nx // 4] = 1  # noqa
        mask[:, ny // 4 : ny // 2, 0 : nx // 2] = 2  # noqa

    logger.info(f"[{well_id}][patched_segment_FOV] END")

    return mask.astype(label_dtype)


def patched_segment_FOV_overlapping_organoids(
    x, label_dtype=None, well_id=None, **kwargs
):

    import logging

    logger = logging.getLogger("cellpose_segmentation.py")
    logger.info(f"[{well_id}][patched_segment_FOV] START")

    # Actual labeling
    mask = np.zeros_like(x)
    nz, ny, nx = mask.shape
    indices = np.arange(0, nx // 2)
    mask[:, indices, indices] = 1  # noqa
    mask[:, indices + 10, indices + 20] = 2  # noqa

    logger.info(f"[{well_id}][patched_segment_FOV] END")

    return mask.astype(label_dtype)


def patched_use_gpu(*args, **kwargs):
    debug("WARNING: using patched_use_gpu")
    return False


def test_failures(
    tmp_path: Path,
    testdata_path: Path,
    zenodo_zarr: List[str],
    zenodo_zarr_metadata: List[Dict[str, Any]],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: MonkeyPatch,
):

    monkeypatch.setattr(
        "fractal_tasks_core.cellpose_segmentation.use_gpu", patched_use_gpu
    )

    monkeypatch.setattr(
        "fractal_tasks_core.cellpose_segmentation.segment_FOV",
        patched_segment_FOV,
    )

    caplog.set_level(logging.WARNING)

    # Use pre-made 3D zarr
    zarr_path = tmp_path / "tmp_out/*.zarr"
    metadata = prepare_3D_zarr(
        str(zarr_path), zenodo_zarr, zenodo_zarr_metadata
    )
    debug(zarr_path)
    debug(metadata)

    # A sequence of invalid attempts
    for component in metadata["image"]:

        kwargs = dict(
            input_paths=[str(zarr_path)],
            output_path=str(zarr_path),
            metadata=metadata,
            component=component,
            level=3,
        )
        # Attempt 1
        cellpose_segmentation(
            **kwargs,
            wavelength_id="invalid_wavelength_id",
        )
        assert "ChannelNotFoundError" in caplog.records[0].msg

        # Attempt 2
        cellpose_segmentation(
            **kwargs,
            channel_label="invalid_channel_name",
        )
        assert "ChannelNotFoundError" in caplog.records[0].msg
        assert "ChannelNotFoundError" in caplog.records[1].msg

        # Attempt 3
        with pytest.raises(ValueError):
            cellpose_segmentation(
                **kwargs,
                wavelength_id="A01_C01",
                channel_label="invalid_channel_name",
            )


def test_workflow_with_per_FOV_labeling(
    tmp_path: Path,
    testdata_path: Path,
    zenodo_zarr: List[str],
    zenodo_zarr_metadata: List[Dict[str, Any]],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: MonkeyPatch,
):

    monkeypatch.setattr(
        "fractal_tasks_core.cellpose_segmentation.use_gpu", patched_use_gpu
    )

    monkeypatch.setattr(
        "fractal_tasks_core.cellpose_segmentation.segment_FOV",
        patched_segment_FOV,
    )

    # Setup caplog fixture, see
    # https://docs.pytest.org/en/stable/how-to/logging.html#caplog-fixture
    caplog.set_level(logging.INFO)

    # Use pre-made 3D zarr
    zarr_path = tmp_path / "tmp_out/*.zarr"
    metadata = prepare_3D_zarr(
        str(zarr_path), zenodo_zarr, zenodo_zarr_metadata
    )
    debug(zarr_path)
    debug(metadata)

    # Per-FOV labeling
    for component in metadata["image"]:
        cellpose_segmentation(
            input_paths=[str(zarr_path)],
            output_path=str(zarr_path),
            metadata=metadata,
            component=component,
            wavelength_id="A01_C01",
            level=3,
            relabeling=True,
            diameter_level0=80.0,
        )

    # OME-NGFF JSON validation
    image_zarr = Path(zarr_path.parent / metadata["image"][0])
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
    zenodo_zarr: List[Path],
    zenodo_zarr_metadata: List[Dict[str, Any]],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: MonkeyPatch,
):
    # Testing by providing the same channel twice as wavelength_id & 
    # wavelength_id_c2

    monkeypatch.setattr(
        "fractal_tasks_core.cellpose_segmentation.use_gpu", patched_use_gpu
    )

    monkeypatch.setattr(
        "fractal_tasks_core.cellpose_segmentation.segment_FOV",
        patched_segment_FOV,
    )

    # Setup caplog fixture, see
    # https://docs.pytest.org/en/stable/how-to/logging.html#caplog-fixture
    caplog.set_level(logging.INFO)

    # Use pre-made 3D zarr
    zarr_path = tmp_path / "tmp_out/*.zarr"
    metadata = prepare_3D_zarr(zarr_path, zenodo_zarr, zenodo_zarr_metadata)
    debug(zarr_path)
    debug(metadata)

    # Per-FOV labeling
    for component in metadata["image"]:
        cellpose_segmentation(
            input_paths=[zarr_path],
            output_path=zarr_path,
            metadata=metadata,
            component=component,
            wavelength_id="A01_C01",
            wavelength_id_c2="A01_C01",
            level=3,
            relabeling=True,
            diameter_level0=80.0,
            model_type="cyto2",
        )

    # OME-NGFF JSON validation
    image_zarr = Path(zarr_path.parent / metadata["image"][0])
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
    zenodo_zarr: List[str],
    zenodo_zarr_metadata: List[Dict[str, Any]],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: MonkeyPatch,
):

    monkeypatch.setattr(
        "fractal_tasks_core.cellpose_segmentation.use_gpu", patched_use_gpu
    )

    # Do not use cellpose
    monkeypatch.setattr(
        "fractal_tasks_core.cellpose_segmentation.segment_FOV",
        patched_segment_FOV,
    )

    # Load pre-made 2D zarr array
    zarr_path_mip = tmp_path / "tmp_out_mip/*.zarr"
    metadata = prepare_2D_zarr(
        str(zarr_path_mip),
        zenodo_zarr,
        zenodo_zarr_metadata,
        remove_labels=True,
    )

    # Per-FOV labeling
    for component in metadata["image"]:
        cellpose_segmentation(
            input_paths=[str(zarr_path_mip)],
            output_path=str(zarr_path_mip),
            metadata=metadata,
            component=component,
            wavelength_id="A01_C01",
            level=2,
            relabeling=True,
            diameter_level0=80.0,
        )

    # OME-NGFF JSON validation
    image_zarr = Path(zarr_path_mip.parent / metadata["image"][0])
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
        "fractal_tasks_core.cellpose_segmentation.use_gpu", patched_use_gpu
    )

    # Do not use cellpose
    monkeypatch.setattr(
        "fractal_tasks_core.cellpose_segmentation.segment_FOV",
        patched_segment_FOV,
    )

    # Init
    img_path = Path(zenodo_images) / "*.png"
    zarr_path = tmp_path / "tmp_out/*.zarr"
    zarr_path_mip = tmp_path / "tmp_out_mip/*.zarr"
    metadata = {}

    # Create zarr structure
    metadata_update = create_ome_zarr(
        input_paths=[str(img_path)],
        output_path=str(zarr_path),
        metadata=metadata,
        allowed_channels=allowed_channels,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        metadata_table="mrf_mlf",
    )
    metadata.update(metadata_update)

    # Yokogawa to zarr
    for component in metadata["image"]:
        yokogawa_to_ome_zarr(
            input_paths=[str(zarr_path)],
            output_path=str(zarr_path),
            metadata=metadata,
            component=component,
        )

    # Replicate
    metadata_update = copy_ome_zarr(
        input_paths=[str(zarr_path)],
        output_path=str(zarr_path_mip),
        metadata=metadata,
        project_to_2D=True,
        suffix="mip",
    )
    metadata.update(metadata_update)
    debug(metadata)

    # MIP
    for component in metadata["image"]:
        maximum_intensity_projection(
            input_paths=[str(zarr_path_mip)],
            output_path=str(zarr_path_mip),
            metadata=metadata,
            component=component,
        )

    # Whole-well labeling
    for component in metadata["image"]:
        cellpose_segmentation(
            input_paths=[str(zarr_path_mip)],
            output_path=str(zarr_path_mip),
            metadata=metadata,
            component=component,
            wavelength_id="A01_C01",
            level=2,
            ROI_table_name="well_ROI_table",
            relabeling=True,
            diameter_level0=80.0,
        )

    # OME-NGFF JSON validation
    image_zarr = Path(zarr_path_mip.parent / metadata["image"][0])
    debug(image_zarr)
    well_zarr = image_zarr.parent
    plate_zarr = image_zarr.parents[2]
    validate_schema(path=str(image_zarr), type="image")
    validate_schema(path=str(well_zarr), type="well")
    validate_schema(path=str(plate_zarr), type="plate")

    check_file_number(zarr_path=image_zarr)


def test_workflow_bounding_box(
    tmp_path: Path,
    testdata_path: Path,
    zenodo_zarr: List[str],
    zenodo_zarr_metadata: List[Dict[str, Any]],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: MonkeyPatch,
):

    monkeypatch.setattr(
        "fractal_tasks_core.cellpose_segmentation.use_gpu", patched_use_gpu
    )

    monkeypatch.setattr(
        "fractal_tasks_core.cellpose_segmentation.segment_FOV",
        patched_segment_FOV,
    )

    # Setup caplog fixture, see
    # https://docs.pytest.org/en/stable/how-to/logging.html#caplog-fixture
    caplog.set_level(logging.INFO)

    # Use pre-made 3D zarr
    zarr_path = tmp_path / "tmp_out/*.zarr"
    metadata = prepare_3D_zarr(
        str(zarr_path), zenodo_zarr, zenodo_zarr_metadata
    )
    debug(zarr_path)
    debug(metadata)

    # Per-FOV labeling
    for component in metadata["image"]:
        cellpose_segmentation(
            input_paths=[str(zarr_path)],
            output_path=str(zarr_path),
            metadata=metadata,
            component=component,
            wavelength_id="A01_C01",
            level=3,
            relabeling=True,
            diameter_level0=80.0,
            bounding_box_ROI_table_name="bbox_table",
        )

    bbox_ROIs = ad.read_zarr(
        str(zarr_path.parent / metadata["image"][0] / "tables/bbox_table/")
    )
    assert bbox_ROIs.shape == (4, 6)
    assert len(bbox_ROIs) > 0
    assert np.max(bbox_ROIs.X) == float(208)


def test_workflow_bounding_box_with_overlap(
    tmp_path: Path,
    testdata_path: Path,
    zenodo_zarr: List[str],
    zenodo_zarr_metadata: List[Dict[str, Any]],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: MonkeyPatch,
):

    monkeypatch.setattr(
        "fractal_tasks_core.cellpose_segmentation.use_gpu", patched_use_gpu
    )

    monkeypatch.setattr(
        "fractal_tasks_core.cellpose_segmentation.segment_FOV",
        patched_segment_FOV_overlapping_organoids,
    )

    # Setup caplog fixture, see
    # https://docs.pytest.org/en/stable/how-to/logging.html#caplog-fixture
    caplog.set_level(logging.INFO)

    # Use pre-made 3D zarr
    zarr_path = tmp_path / "tmp_out/*.zarr"
    metadata = prepare_3D_zarr(
        str(zarr_path), zenodo_zarr, zenodo_zarr_metadata
    )
    debug(zarr_path)
    debug(metadata)

    # Per-FOV labeling
    for component in metadata["image"]:
        with pytest.raises(ValueError):
            cellpose_segmentation(
                input_paths=[str(zarr_path)],
                output_path=str(zarr_path),
                metadata=metadata,
                component=component,
                wavelength_id="A01_C01",
                level=3,
                relabeling=True,
                diameter_level0=80.0,
                bounding_box_ROI_table_name="bbox_table",
            )
